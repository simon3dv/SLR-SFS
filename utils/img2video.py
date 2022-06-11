import os
src_dir = "" # this directory contains many subdirectorys, each subdirectorys contain N imgs
tgt_dir = "PredImg"
if not os.path.exists(tgt_dir): os.makedirs(tgt_dir)
names = os.listdir(src_dir)
for i, name in enumerate(names):
    os.system("ffmpeg -i %s/%s/PredImg/%%06d.png PredImg/%s.gif"%(src_dir,name,name))