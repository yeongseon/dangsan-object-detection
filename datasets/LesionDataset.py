'''
    Author: Wonseon Lim
'''
import os, os.path as osp
import ast
from io import BytesIO
import base64
from glob import glob
from tqdm import tqdm
import json
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split, KFold

import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from torchvision import transforms as ts

import datasets.transforms as T # for test

from utils.logger import init_logger

log_level = "DEBUG"
logger = init_logger("Dataloader", log_level)

class LesionDataset(Dataset):
    def __init__(self,
                 data_dir, 
                 augmentation=None,
                 transforms=None,
                 mode='train'):
        """
        Args:
            data_dir (string): Directory with all train and test dataset
            mode (string): 'train', 'valid' or 'test' mode
        Output:
            boxes ()
        """
        self.mode = mode
        if augmentation is not None : self.augmentation = augmentation
        self.transforms = transforms
        json_list = self.load_json_list(data_dir)
        self.file_name = [json_file['file_name'] for json_file in json_list]

        if mode == 'train' or 'valid':
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
        
    def load_json_list(self, data_dir):
        if self.mode == 'train':
            train_files = sorted(glob(data_dir + '/train/*'))
            train_files, _ = train_test_split(train_files, train_size=0.8, random_state=0, shuffle=True)
            json_list = []
            print('Train set is loading')
            for file in tqdm(train_files):
                with open(file, "r") as json_file:
                    json_list.append(json.load(json_file))
        elif self.mode == 'valid':
            train_files = sorted(glob(data_dir + '/train/*'))
            _, valid_files = train_test_split(train_files, test_size=0.2, random_state=0, shuffle=True)
            json_list = []
            print('Valid set is loading')
            for file in tqdm(valid_files):
                with open(file, "r") as json_file:
                    json_list.append(json.load(json_file))
        else:
            test_files = sorted(glob(data_dir + '/test/*'))
            json_list = []
            print('Test set is loading')
            for file in tqdm(test_files):
                with open(file, "r") as json_file:
                    json_list.append(json.load(json_file))

        return json_list

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):

        file_name = self.file_name[idx]

        try:
            img = Image.open(BytesIO(base64.b64decode(self.imgs[idx]))).convert("RGB")
            img.verify()
            new_h, new_w, _ = np.array(img).shape
        except (IOError, SyntaxError) as e:
            # logger.warning('Bad file:', file_name)
            print(('Bad file:', file_name))

        # img = self.transforms(img)
        
        target = {}
        if self.mode == 'train' or 'valid':
            boxes = []
            for point in self.points[idx]:
                x_min = int(np.min(np.array(point)[:,0]))
                x_max = int(np.max(np.array(point)[:,0]))
                y_min = int(np.min(np.array(point)[:,1]))
                y_max = int(np.max(np.array(point)[:,1]))
                boxes.append([x_min, y_min, x_max, y_max])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            label = [self.label_map[label] for label in self.labels[idx]]

            masks = []
            for box in boxes:
                mask = np.zeros([int(self.heights[idx]), int(self.widths[idx])], np.uint8)
                masks.append(cv2.rectangle(mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 1, -1))

            masks = torch.tensor(masks, dtype=torch.uint8)

            target["boxes"] = boxes
            target["labels"] = torch.tensor(label, dtype=torch.int64)
            target["masks"] = masks
            target["area"] = area
            target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx], dtype=torch.int64)
        target["img_size"] = (new_h, new_w)
        target["img_scale"] = torch.tensor([1.0])

        if self.mode == 'test':
            target["file_name"] = file_name

        # transformations using albumentation library
        if self.transforms is not None:
            if len(target['boxes']) != 0:
                transformed = self.transforms(image=np.array(img),
                                              bboxes=target['boxes'],
                                              class_labels=target['labels'])
                img = torch.as_tensor(transformed['image'])
                target["boxes"] = torch.as_tensor(transformed['bboxes'],
                                                  dtype=torch.float32)
                target["labels"] = torch.as_tensor(transformed['class_labels'])
            else:  # negative samples
                if self.mode == 'train' or 'valid':
                    transforms = []
                    transforms.append(A.Resize(width=self.augmentation.size.w, height=self.augmentation.size.h))
                    transforms.append(A.HorizontalFlip(p=0.5))
                    transforms.append(A.VerticalFlip(p=0.5))
                    transforms.append(A.RandomBrightnessContrast(p=0.6))
                    transforms.append(ts.ToTensor())
                    transforms = A.Compose(transforms)
                    transformed = transforms(image=np.array(img))
                    img = torch.as_tensor(transformed['image'])
                else:
                    transforms = []
                    transforms.append(A.Resize(width=self.augmentation.size.w, height=self.augmentation.size.h))
                    transforms.append(ts.ToTensor())
                    transforms = A.Compose(transforms)
                    transformed = transforms(image=np.array(img))
                    img = torch.as_tensor(transformed['image'])

        return img, target

# def collate_fn(batch):
#     return tuple(zip(*batch))

def collate_fn(batch):
        images, targets = tuple(zip(*batch))
        # images = torch.stack(images)
        # images = images.float()


        boxes = [target["boxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        image_ids = [target["image_id"] for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "boxes": boxes,
            "labels": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets, image_ids



if __name__ == "__main__":

    import yaml
    from easydict import EasyDict
    config_file = './dangsan-object-detection/config/config.yaml'
    with open(config_file, 'r') as stream:
            try:
                config = EasyDict(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)
    format = "pascal_voc" 

    train_dataset = LesionDataset(
            data_dir='./data',
            augmentation=config.augmentation,
            transforms=T.get_transform(
                True, config.augmentation, format=format
            ),  # ?FIXME changer format en fonction de fasterRCNN et yolo
            mode='train',
        )

    # train_dataset = LesionDataset(data_dir='./data', 
    #                                mode='train')
    torch.manual_seed(1)
    # indices = torch.randperm(len(train_dataset)).tolist()
    # train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=0,
        collate_fn=collate_fn)

    for data, annot, targets, image_ids in train_data_loader:
        data.shape
        break