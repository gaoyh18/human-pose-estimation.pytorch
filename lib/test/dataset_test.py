from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import copy
import random
import json_tricks as json
import numpy as np

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)

class testdataset(Dataset):
    def __init__(self, cfg, imagepath, is_train, transform=None):
    
        self.pixel_std = 200
        self.imagepath=imagepath
        self.is_train=is_train
        self.transform = transform
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.image_size = cfg.MODEL.IMAGE_SIZE
        
        
    def _load_coco_keypoint_annotation_kernal(self):   #返回rec，{}包括这张图片的路径，单人clear_bbox信息
        
        pic=cv2.imread(self.imagepath, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        size=pic.shape
        
        width = float(size[1])#width.dtype=np.float32
        height = float(size[0])#height.dtype=np.float32

        center, scale = self._xywh2cs(0,0,width,height)
        rec = []
        rec.append({
            'image': self.imagepath,
            'center': center,                            #center=中心
            'scale': scale,                              #scale=(w,h)
            'filename': '',
            'imgnum': 0,
        })
        return rec

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:  #宽高比
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],   #除以200
            dtype=np.float32)


        return center, scale
    
    def getitem(self):    #return input, meta
        db_rec =self._load_coco_keypoint_annotation_kernal() 
        image_file = db_rec[0]['image']
        filename = db_rec[0]['filename'] if 'filename' in db_rec[0] else ''
        imgnum = db_rec[0]['imgnum'] if 'imgnum' in db_rec[0] else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(  #(文件名，标记)
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))


        c = db_rec[0]['center']
        s = db_rec[0]['scale']
        score = db_rec[0]['score'] if 'score' in db_rec[0] else 1
        r = 0

        trans = get_affine_transform(c, s, r, self.image_size)  #将s经过r缩放到imagesize   矩阵2x3
        input = cv2.warpAffine(                #input是一个旋转缩放过的原图
            data_numpy,  #输入图像      
            trans,       #变换矩阵
            (int(self.image_size[0]), int(self.image_size[1])),  #输出图像大小192x256
            flags=cv2.INTER_LINEAR)   #插值方法

        if self.transform:
            input = self.transform(input)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'center': c,
            'scale': s,
            'score': score,
            'joints_vis':[[(1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0),(1,1,0)]]
        }

        return input, meta