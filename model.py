import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def mask_rcnn(num_classes, pretrained=False, pretrained_backbone=False, min_size=512, max_size=1024):
    # Load Mask R-CNN with optional pretraining
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=pretrained, pretrained_backbone=pretrained_backbone, min_size=min_size, max_size=max_size)

    # Replace box predictor
    in_channels_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels_box, num_classes)

    # Replace mask predictor
    in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels_mask, 256, num_classes)

    return model
