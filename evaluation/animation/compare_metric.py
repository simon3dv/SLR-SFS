import os
import cv2
import numpy as np
import time
import sys

if len(sys.argv) > 1:
    json1 = sys.argv[1]
else:
    json1 = "metric1.json"
if len(sys.argv) > 2:
    json2 = sys.argv[2]
else:
    json2 = 'metric2.json'

import json
with open(json1, 'r') as load_f:
  label1 = json.load(load_f)
with open(json2, 'r') as load_f:
  label2 = json.load(load_f)

for i, x in enumerate( sorted(label1["LPIPS"].keys() )):
    score1 = label1["LPIPS"][x]
    score2 = label2["LPIPS"][x]
    if score1 < score2:
        print("ours better %s %f"%(x, score2-score1))


for i, x in enumerate( sorted(label1["LPIPS"].keys() )):
    score1 = label1["LPIPS"][x]
    score2 = label2["LPIPS"][x]
    if score1 > score2:
        print(" better %s %f"%(x, score1-score2))
