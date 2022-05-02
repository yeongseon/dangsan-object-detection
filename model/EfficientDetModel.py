import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import traceback
import torchvision.models as models

from effdet import * # get_efficientdet_config
from effdet.efficientdet import HeadNet
from effdet.anchors import Anchors, AnchorLabeler, generate_detections
from effdet.loss import DetectionLoss
from effdet.config import set_config_readonly, set_config_writeable

from effdet import create_model

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
class EfficientDetModel(nn.Module): 
    def __init__(self,
                 nclasses: int=2,
                 image_size: int=512, # size % 128 = 0
                 pretrained=False,
                 pretrained_backbone=False) -> None:
        super(EfficientDetModel, self).__init__()

        self.model = create_model('tf_efficientdet_d1',
                                  bench_task='train',
                                  num_classes=nclasses,
                                  image_size=(image_size, image_size),
                                  bench_labeler=True,
                                  pretrained=pretrained)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Output during training:
        - loss_dict (``dict[loss, class_loss, box_loss]``) : the calculated losses

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
