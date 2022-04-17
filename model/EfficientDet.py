import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import traceback
import torchvision.models as models
from efficientdet_pytorch.effdet import * # get_efficientdet_config
from efficientdet_pytorch.effdet.efficientdet import HeadNet
from efficientdet_pytorch.effdet.anchors import Anchors, AnchorLabeler, generate_detections
from efficientdet_pytorch.effdet.loss import DetectionLoss
from efficientdet_pytorch.effdet.config import set_config_readonly, set_config_writeable

def get_efficientDet(nclasses, 
                     image_size,
                     pretrained, # dummy for template
                     pretrained_backbone):
    config = get_efficientdet_config('tf_efficientdet_d1')
    net = EfficientDet(config, pretrained_backbone=pretrained_backbone)
    set_config_writeable(config)
    config.num_classes = nclasses
    config.image_size = image_size
    set_config_readonly(config)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    return net, config

# class name EfficientDetModel
class EfficientDet(nn.Module): 
    def __init__(self,
                 nclasses: int=2,
                 image_size: int=512, # size % 128 = 0
                 pretrained=False,
                 pretrained_backbone=False) -> None:
        super(EfficientDet, self).__init__()

        self.model, self.config = get_efficientDet(nclasses, 
                                                   image_size, 
                                                   pretrained, 
                                                   pretrained_backbone)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Output during inference:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
            between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction 
        """
        if y != None:
            return self.model(x, y)
        else:
            return self.model(x)
