"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-

import sys
import os
import time
import argparse
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict
import config
from text_recognition.demo import text_recognition
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def test_net(net, image, text_threshold, link_threshold, low_text):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, config.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=config.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    # polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    # for k in range(len(polys)):
    #     if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    #render_img = score_text.copy()
    #render_img = np.hstack((render_img, score_link))
    #ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes

if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + config.craft_model + ')')
    net.load_state_dict(copyStateDict(torch.load(config.craft_model, map_location='cpu')))

    net.eval()
    ocr_result=[]
    #input_image = request.files['image']
    input_image_path = "test_images/4.png"
    image = cv2.imread(input_image_path)
    t = time.time()
    bboxes = test_net(net, image, config.text_threshold, config.link_threshold, config.low_text)
    for box in bboxes:
        poly = np.array(box).astype(np.int32).reshape((-1))#poly=["x1","y1","x2",y1,x2,"y2",x1,y2]

        x = poly[0]
        y = poly[1]
        x_w = poly[2]
        y_h = poly[5]
        cropped_img = image[y:y_h, x:x_w]
        ocr_result.append(text_recognition(cropped_img))
    print("elapsed time : {}s".format(time.time() - t))
    print(ocr_result)
    file_utils.saveResult(input_image_path, image, bboxes, dirname='result/')


