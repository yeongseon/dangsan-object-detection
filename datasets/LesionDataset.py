# -*- coding: utf-8 -*-
'''
    Author: Clément APAVOU
'''
import os, os.path as osp
import ast
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from .transforms import ToTensor
from torchvision import transforms

from utils.logger import init_logger
from io import BytesIO
import base64
import cv2

import json

log_level = "DEBUG"
logger = init_logger("Dataloader", log_level)


class LesionDataset(Dataset):
    def __init__(self, json_list, mode='train'):
        self.mode = mode
        self.file_name = [json_file['file_name'] for json_file in json_list]
        if mode == 'train':
            self.labels = []
            for data in json_list:
                label = []
                for shapes in data['shapes']:
                    label.append(shapes['label'])
                self.labels.append(label)
            self.points = []
            for data in json_list:
                point = []
                for shapes in data['shapes']:
                    point.append(shapes['points'])
                self.points.append(point)
        self.imgs = [data['imageData'] for data in json_list]
        
        self.widths = [data['imageWidth'] for data in json_list]
        self.heights = [data['imageHeight'] for data in json_list]
        
        self.label_map ={
            '01_ulcer':1, '02_mass':2, '04_lymph':3, '05_bleeding':4
        }
        
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p = 0.5),
            #transforms.RandomCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600))
        ])

        self.transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600))
        ])
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        file_name = self.file_name[i]
        img = Image.open(BytesIO(base64.b64decode(self.imgs[i])))
        
        target = {}
        if self.mode == 'train':
            img = self.transforms(img)
            boxes = []
            for point in self.points[i]:
                x_min = int(np.min(np.array(point)[:,0]))
                x_max = int(np.max(np.array(point)[:,0]))
                y_min = int(np.min(np.array(point)[:,1]))
                y_max = int(np.max(np.array(point)[:,1]))
                boxes.append([x_min, y_min, x_max, y_max])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            label = [self.label_map[label] for label in self.labels[i]]

            masks = []
            for box in boxes:
                mask = np.zeros([int(self.heights[i]), int(self.widths[i])], np.uint8)
                masks.append(cv2.rectangle(mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 1, -1))

            masks = torch.tensor(masks, dtype=torch.uint8)

            target["boxes"] = boxes
            target["labels"] = torch.tensor(label, dtype=torch.int64)
            target["masks"] = masks
            target["area"] = area
            target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([i], dtype=torch.int64)
        if self.mode == 'test':
            img = self.transforms_test(img)
            target["file_name"] = file_name
            
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))

def json2list(train_files):
        
    train_json_list = []
    for file in train_files:
        with open(file, "r") as json_file:
            train_json_list.append(json.load(json_file))
    return train_json_list

