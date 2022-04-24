# -*- coding: UTF-8 -*-
import sys
import os
import glob
import json
if __name__=='__main__':
    rootdir = "data/eulerian_data/validation"
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = rootdir
    if len(sys.argv) > 2:
        flow_dir = sys.argv[2]
    else:
        flow_dir = rootdir
    if len(sys.argv) > 3:
        save_dir = sys.argv[3]
    else:
        save_dir = os.path.join(rootdir+'_output')

    if len(sys.argv) > 4:
        warp_pretrained_model = sys.argv[4]
    else:
        warp_pretrained_model = ''
    if len(sys.argv) > 5:
        name = sys.argv[5]
    else:
        name = "Demo"
    if len(sys.argv) > 6:
        W = int(sys.argv[6])
    else:
        W = 512
    if len(sys.argv) > 7:
        N = int(sys.argv[7])
    else:
        N = 150

    if len(sys.argv) > 8:
        speed = float(sys.argv[8])
    else:
        speed = 0.25

    if len(sys.argv) > 9:
        PYFILE = sys.argv[9]
    else:
        PYFILE = "test.py"

    if len(sys.argv) > 10:
        SCENE = sys.argv[10]
    else:
        SCENE = "00739"

    if len(sys.argv) > 11:
        ALIGNFILE = sys.argv[11]
    else:
        ALIGNFILE = "align_max_frame.json"

    if len(sys.argv) > 12:
        start_index = int(sys.argv[12])
    else:
        start_index = -1

    if len(sys.argv) > 13:
        end_index = int(sys.argv[13])
    else:
        end_index = -1
    if os.path.exists(ALIGNFILE):
        with open(ALIGNFILE, "r") as f:
            align_dict = json.load(f)
    scenes = glob.glob(os.path.join(image_dir,"*_input.jpg"))
    for i_scene, scene_file in enumerate(sorted(scenes)):
        if start_index != -1 and i_scene < start_index:
            continue
        if end_index != -1 and i_scene > end_index:
            continue
        SCENE = scene_file[-21:-10]
        if ALIGNFILE != "None" and SCENE not in align_dict.keys():
            continue
        images = glob.glob(os.path.join(image_dir,SCENE+"*_input.jpg"))
        print("Processing {}".format(images))
        for i_name, image_file in enumerate(sorted(images)): #and '00000' in x])):
            flow_file = image_file.replace("input.jpg", "motion.pth")
            name = image_file[-21:-10]
            print("Processing {}".format(name))
            os.system("python test_animating/{} {} {} {} '{}' {} {} {} {} {}".format(
                PYFILE,
                image_file,
                flow_file,
                os.path.join(save_dir, name),
                warp_pretrained_model,
                name,
                W,
                N,
                speed,
                ALIGNFILE))
