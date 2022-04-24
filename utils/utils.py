import os
import sys
import shutil
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import torch
import av
import lz4framed
import pickle
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 1e-14

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_midas_flowsize(depth_path, W, centercrop=False):
    depth = Image.open(depth_path)
    depth = transforms.CenterCrop((1024, 1920))(depth)
    if centercrop:
        depth = transforms.CenterCrop((1024, 1024))(depth)
    depth = transforms.Resize((W, W), 0)(depth)
    depth = np.asarray(depth).astype(np.float32)
    depth = depth - depth.min()
    depth = cv2.blur(depth, ksize=(3, 3))
    depth = depth / depth.max()#[0,1]
    depth = 1. / (depth * 10 + 0.01)
    #depth[depth == 0] = -1
    depth = torch.from_numpy(depth).unsqueeze(0)
    return depth


def load_midas(depth_path):
    depth = Image.open(depth_path)
    depth = np.asarray(depth).astype(np.float32)
    depth = depth - depth.min()
    depth = cv2.blur(depth, ksize=(3, 3))
    depth = depth / depth.max()#[0,1]
    depth = 1. / (depth * 10 + 0.01)
    #depth[depth == 0] = -1
    depth = torch.from_numpy(depth).unsqueeze(0)
    return depth

def load_depth(depth_path, W, disp_rescale=10, preprocess = True):
    depth = Image.open(depth_path)
    if preprocess:
        w,h = depth.size
        depth = transforms.CenterCrop(min(h,w))(depth)
        depth = transforms.Resize((W, W), 0)(depth)
    depth = np.asarray(depth)
    depth = depth - depth.min()
    depth = cv2.blur(depth / depth.max(), ksize=(3, 3)) * depth.max()
    depth = (depth / depth.max()) * disp_rescale
    #if h is not None and w is not None:
    #    depth = resize(depth / depth.max(), (h, w), order=1) * depth.max()
    depth = 1. / np.maximum(depth, 0.05)
    depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
    return depth


def path_planning(num_frames, pose_range, path_type=''):
    pose_range = np.array(pose_range, dtype='float32').reshape(-1)
    if path_type == 'straight-line':
        t = np.linspace(0, 1, num_frames)
        pose = np.concatenate(( \
            np.zeros((num_frames, 3)), \
            t.reshape((-1, 1)).dot(pose_range.reshape((1, -1)))), 1)
    elif path_type == 'circle':
        t = np.arange(-2., 2., 4. / num_frames).reshape((-1, 1)) * np.pi
        pose = np.concatenate(( \
            np.zeros((num_frames, 3)), \
            np.cos(t) * pose_range[0], \
            np.sin(t) * pose_range[1], \
            np.cos(t / 2) * pose_range[2]), 1)
    elif path_type == 'round' or path_type == 'cone':
        t = np.arange(0, 2., 2. / num_frames) * np.pi
        u = np.cos(t) * np.tan(pose_range[0] / 2)
        v = np.sin(t) * np.tan(pose_range[1] / 2)
        if path_type == 'round':
            u *= t / (2 * np.pi)
            v *= t / (2 * np.pi)
        n = u * u + v * v
        x = 2 * u / (1 + n)
        y = 2 * v / (1 + n)
        z = (1 - n) / (1 + n)
        nx = np.vstack((1 - x * x / (1 + z), -x * y / (1 + z), -x)).T
        ny = np.vstack((-x * y / (1 + z), 1 - y * y / (1 + z), -y)).T
        nz = np.vstack((x, y, z)).T

        pose = np.zeros((num_frames, 6))
        for i in range(num_frames):
            R = np.vstack((nx[i], ny[i], nz[i]))
            pose[i, :3] = cv2.Rodrigues(R)[0].reshape(-1)
            pose[i, 3:] = R.dot([x[i], y[i], z[i] - 1]) * pose_range[2]
    return pose

def load_compressed_tensor(filename):
    retval = None
    with open(filename, mode='rb') as file:
        retval = torch.from_numpy(pickle.loads(lz4framed.decompress(file.read())))
    return retval

class VideoReader(object):
    """
    Wrapper for PyAV

    Reads frames from a video file into numpy tensors. Example:

    file = VideoReader(filename)
    video_frames = file[start_frame:end_frame]

    If desired, a table-of-contents (ToC) file can be provided to speed up the loading/seeking time.
    """

    def __init__(self, file, toc=None, format=None):
        if not hasattr(file, 'read'):
            file = str(file)
        self.file = file
        self.format = format
        self._container = None

        with av.open(self.file, format=self.format) as container:
            stream = [s for s in container.streams if s.type == 'video'][0]
            self.bit_rate = stream.bit_rate

            # Build a toc
            if toc is None:
                packet_lengths = []
                packet_ts = []
                for packet in container.demux(stream):
                    if packet.stream.type == 'video':
                        decoded = packet.decode()
                        if len(decoded) > 0:
                            packet_lengths.append(len(decoded))
                            packet_ts.append(decoded[0].pts)
                self._toc = {
                    'lengths': packet_lengths,
                    'ts': packet_ts,
                }
            else:
                self._toc = toc

            self._toc_cumsum = np.cumsum(self.toc['lengths'])
            self._len = self._toc_cumsum[-1]

            # PyAV always returns frames in color, and we make that
            # assumption in get_frame() later below, so 3 is hardcoded here:
            self._im_sz = stream.height, stream.width, 3
            self._time_base = stream.time_base
            self.rate = stream.average_rate

        self._load_fresh_file()

    @staticmethod
    def _next_video_packet(container_iter):
        for packet in container_iter:
            if packet.stream.type == 'video':
                decoded = packet.decode()
                if len(decoded) > 0:
                    return decoded

        raise ValueError("Could not find any video packets.")

    def _load_fresh_file(self):
        if self._container is not None:
            self._container.close()

        if hasattr(self.file, 'seek'):
            self.file.seek(0)

        self._container = av.open(self.file, format=self.format)
        demux = self._container.demux(self._video_stream)
        self._current_packet = self._next_video_packet(demux)
        self._current_packet_no = 0

    @property
    def _video_stream(self):
        return [s for s in self._container.streams if s.type == 'video'][0]

    def __len__(self):
        return self._len

    def __del__(self):
        if self._container is not None:
            self._container.close()

    def __getitem__(self, item):
        if isinstance(item, int):
            item = slice(item, item + 1)

        if item.start < 0 or item.start >= len(self):
            raise IndexError(f"start index ({item.start}) out of range")

        if item.stop < 0 or item.stop > len(self):
            raise IndexError(f"stop index ({item.stop}) out of range")

        return np.stack([self.get_frame(i) for i in range(item.start, item.stop)])

    @property
    def frame_shape(self):
        return self._im_sz

    @property
    def toc(self):
        return self._toc

    def get_frame(self, j):
        # Find the packet this frame is in.
        packet_no = self._toc_cumsum.searchsorted(j, side='right')
        self._seek_packet(packet_no)
        # Find the location of the frame within the packet.
        if packet_no == 0:
            loc = j
        else:
            loc = j - self._toc_cumsum[packet_no - 1]
        frame = self._current_packet[loc]  # av.VideoFrame

        return frame.to_ndarray(format='rgb24')

    def _seek_packet(self, packet_no):
        """Advance through the container generator until we get the packet
        we want. Store that packet in selfpp._current_packet."""
        packet_ts = self.toc['ts'][packet_no]
        # Only seek when needed.
        if packet_no == self._current_packet_no:
            return
        elif (packet_no < self._current_packet_no
              or packet_no > self._current_packet_no + 1):
            self._container.seek(packet_ts, stream=self._video_stream)

        demux = self._container.demux(self._video_stream)
        self._current_packet = self._next_video_packet(demux)
        while self._current_packet[0].pts < packet_ts:
            self._current_packet = self._next_video_packet(demux)

        self._current_packet_no = packet_no

def read_flo(strFile):
    with open(strFile, 'rb') as objFile:
        strFlow = objFile.read()

    assert(np.frombuffer(buffer=strFlow, dtype=np.float32, count=1, offset=0) == 202021.25)

    intWidth = np.frombuffer(buffer=strFlow, dtype=np.int32, count=1, offset=4)[0]
    intHeight = np.frombuffer(buffer=strFlow, dtype=np.int32, count=1, offset=8)[0]

    return np.frombuffer(buffer=strFlow, dtype=np.float32, count=intHeight * intWidth * 2, offset=12).reshape([ intHeight, intWidth, 2 ])