# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, render_template

from flask_restful import Api
import numpy as np
import io
import json
import config
import time, os
app = Flask(__name__)
api = Api(app)


import sys
import argparse
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

net = CRAFT()  # initialize

print('Loading weights from checkpoint (' + config.craft_model + ')')
net.load_state_dict(copyStateDict(torch.load(config.craft_model, map_location='cpu')))

net.eval()
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

    t1 = time.time() - t1



    print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes


# flask 이미지 모델입력에 적합한 형식으로 바꾸기
def request_file_to_image(f_read, save_image_on=True):
    nparr = np.frombuffer(f_read, np.uint8)

    # IMREAD_UNCHANGED #IMREAD_COLOR
    original_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    # original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    if original_img.shape[2] > 3:
        original_img_3ch = original_img[:, :, :3]
    else:
        original_img_3ch = original_img  # [:,:,:3]

    try:
        # save post image
        current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
        print('###################')
        img_save_path = "{}/{}".format('input', current_time+'.png' )
        print(img_save_path)
        cv2.imwrite(img_save_path, original_img)


    except Exception as e:
        print("e :{}\nCould not save image".format(e))

    return original_img_3ch
@app.route('/',methods=['POST', 'GET'])
def index_page():
    return render_template('index.html')

@app.route('/ocr', methods=['POST','GET'])
def ocr():
    """Regist user feature data on database and return complete or not

    Request:
        user_id         (str)   : request user_id from smartphone or tablet
        palm_feature    (json)  : request palm_feature from smartphone or tablet
        gap_features    (json)  : request gap_features from smartphone or tablet (index, middle, ring, pinky)
        model_version   (str)   : request model_version from smartphone or tablet

    Returns:
        json(dict): result_code, result_message
    """
    ocr_output_dict = {}
    ocr_result=[]
    start = time.time()
    # load net

    #input_image = request.files['image']
    if request.form:
        ocr_output_dict["user_id"] = request.form.get('user_id')
        ocr_output_dict["user_name"] = request.form.get('user_name')

    f = request.files['image']

    f_read = f.read()
    image = request_file_to_image(f_read, save_image_on=False)

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
    input_image_path = "test_images/4.png"
    file_utils.saveResult(input_image_path, image, bboxes, dirname='result/')

    print("ocr time :", time.time() - start)

    ocr_output_dict["score_info"] = {'club_name': ocr_result}
    ocr_output_dict["result_code"] = 0

    return jsonify(ocr_output_dict)

if __name__ == '__main__':
    app.run(debug=True, host='10.12.200.137', port=5000)
