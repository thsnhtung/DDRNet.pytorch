import base64
import numpy as np
from numpy.lib.type_check import imag
import socketio
import eventlet
import eventlet.wsgi
import cv2
from PIL import Image
from flask import Flask
from io import BytesIO


#------------- Add library ------------#
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import math


import _init_paths
import torch
import torch.nn as nn
from PIL import Image


import argparse
import os
import pprint
import sys

import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, speed_test

#--------------------------------------#

#  model = FastSCNN(2, **kwargs)
#     if pretrained:
#         if(map_cpu):
#             model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset]), map_location='cpu'))
#         else:
#             model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset])))
#     return model



def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/map/map_hrnet_ocr_w18_small_v2_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def input_transform(image):
    image = image[90:170,:]
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).to('cuda')
    return image





#out = cv2.VideoWriter('C:\\Users\\Asus\\Desktop\\AICAR\\Unity-UIT_Car\\Code test Simulation\\Raw code\\Video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, (320,180))
#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    global lane_model, speed_model, i, origin_img
    if data:
        steering_angle = 0  #Góc lái hiện tại của xe
        speed = 0           #Vận tốc hiện tại của xe
        image = 0           #Ảnh gốc

        steering_angle = float(data["steering_angle"])
        speed = float(data["speed"])
        #Original Image
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        origin_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


        image = input_transform(origin_img)


        """
        - Chương trình đưa cho bạn 3 giá trị đầu vào:
            * steering_angle: góc lái hiện tại của xe
            * speed: Tốc độ hiện tại của xe
            * image: hình ảnh trả về từ xe
        
        - Bạn phải dựa vào 3 giá trị đầu vào này để tính toán và gửi lại góc lái và tốc độ xe cho phần mềm mô phỏng:
            * Lệnh điều khiển: send_control(sendBack_angle, sendBack_Speed)
            Trong đó:
                + sendBack_angle (góc điều khiển): [-25, 25]  NOTE: ( âm là góc trái, dương là góc phải)
                + sendBack_Speed (tốc độ điều khiển): [-150, 150] NOTE: (âm là lùi, dương là tiến)
        """
        sendBack_angle = 0
        sendBack_Speed = 0
        try:
            #------------------------------------------  Work space  ----------------------------------------------#
           
            with torch.no_grad():
                out = model(image)
                

                out = F.interpolate(
                    out[0], (80, 320), 
                    mode='bilinear', align_corners= True
                )   

                _, pred = torch.max(out, dim=1)
                pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)

                mask = np.zeros((180, 320), dtype=np.uint8)
                mask[90:170] = pred
                color = np.array([0,255,0], dtype='uint8')
               
                masked_img = np.where(mask[...,None], color, origin_img)

                # use `addWeighted` to blend the two images
                # the object will be tinted toward `color`
                pred_image = cv2.addWeighted(origin_img, 0.5, masked_img, 0.5,0)

                # pred_image = origin_img[90:170, :, :]
                # contours, hierarchy = cv2.findContours(image=pred, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                
                # print(contours)
                # cv2.drawContours(image=pred_image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

                cv2.imshow('image ', pred_image)
                cv2.waitKey(1)
          
            #------------------------------------------------------------------------------------------------------#
            #print('{} : {}'.format(sendBack_angle, sendBack_Speed))
            send_control(sendBack_angle, sendBack_Speed)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
        },
        skip_sid=True)



if __name__ == '__main__':
    # use this to change which GPU to use

# set the modified tf session as backend in keras

    #-----------------------------------  Setup  ------------------------------------------#
   
    # use this to change which GPU to use

# set the modified tf session as backend in keras
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        # model_state_file = os.path.join(final_output_dir, 'best_0.7589.pth')
        model_state_file = os.path.join(final_output_dir, 'best.pth')    
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    #--------------------------------------------------------------------------------------#
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)