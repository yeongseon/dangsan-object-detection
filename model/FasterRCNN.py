import torch
import torch.nn as nn
import torchvision.models as models

class FasterRCNN(nn.Module):
    """
    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
    https://arxiv.org/abs/1506.01497
    
    Pre-trained
    https://pytorch.org/vision/main/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
    """
    def __init__(self,
                 nclasses: int = 2,
                 pretrained=False,
                 pretrained_backbone=False) -> None:
        super(FasterRCNN, self).__init__()

        model = models.detection.fasterrcnn_resnet50_fpn(
            pretrained=pretrained, pretrained_backbone=pretrained_backbone)
        #
        # Faster r-cnn with ResNet-50
        # pretrained(bool): If True, returns a model pre-trained on COCO train2017
        # pretrained_backbone(bool): If True, returns a model with backbone pre-trained on Imagenet
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, nclasses)
        # in_channel (int) : number of input channels
        # num_classes (int) : number of output classes
        # return classification, bounding box regression

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
