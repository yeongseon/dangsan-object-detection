"""
Written by BH
"""
import torch
import torch.nn as nn
import torchvision.models as models
import math

class RetinaNet(nn.Module):
    """
    nclasses: 분류해야 하는 Class 수
    Pretrained 
    
    """
    def __init__(self,
                 nclasses: int = 2,
                 pretrained=False,
                 pretrained_backbone=False) -> None:

        super(RetinaNet, self).__init__()

        """
        https://pytorch.org/vision/main/generated/torchvision.models.detection.retinanet_resnet50_fpn.html
        https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py

        pretrained(bool) = True → Return model pre-trained on COCO train2017 
        pretrained_backbone(bool) = True → Return a model with backbone pre-trained on Imagenet
        """
        model = models.detection.retinanet_resnet50_fpn(
            pretrained=pretrained, pretrained_backbone=pretrained_backbone)

        #nuumber of anchors to be predicted
        num_anchors = model.head.classification_head.num_anchors
        # __init__ function
        model.head.classification_head.num_classes = nclasses
        out_features = model.head.classification_head.conv[0].out_channels
        # parameters: in_channel, out_channel, kernenl size
        cls_logits = torch.nn.Conv2d(out_features, num_anchors * nclasses, kernel_size = 3, stride=1, padding=1)
        # Initialized Tensor with normal distribution
        torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
        # Initialized Tensor Bias with normal distribution (Logit Function)
        torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code 
        # assign cls head to model
        model.head.classification_head.cls_logits = cls_logits

        self.model = model

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
