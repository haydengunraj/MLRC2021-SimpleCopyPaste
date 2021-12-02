from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor, maskrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection.rpn import AnchorGenerator

from data import NUM_CLASSES


def maskrcnn_from_config(config, override_pretraining=False):
    """Helper for building a Mask R-CNN model"""
    if override_pretraining:
        config['pretrained'] = False
        config['pretrained_backbone'] = False
    return maskrcnn(
        config['backbone'], NUM_CLASSES, pretrained=config['pretrained'],
        pretrained_backbone=config['pretrained_backbone'],
        min_size=config['min_size'], max_size=config['max_size'])


def maskrcnn(backbone_name, num_classes, pretrained=False, pretrained_backbone=False, min_size=1024, max_size=2048):
    """Builds a Mask R-CNN model with a ResNet-50 FPN or MobileNetV2 FPN backbone"""
    if backbone_name == 'resnet50_fpn':
        # Build Mask R-CNN ResNet-50 FPN
        model = maskrcnn_resnet50_fpn(
            pretrained=pretrained, pretrained_backbone=pretrained_backbone, min_size=min_size, max_size=max_size)

        # Replace box predictor
        in_channels_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_channels_box, num_classes)

        # Replace mask predictor
        in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels_mask, 256, num_classes)
    elif backbone_name in ('mobilenet_v2_fpn', 'mobilenet_v3_large_fpn', 'mobilenet_v3_small_fpn'):
        # Build Mask R-CNN MobileNet FPN
        model = _maskrcnn_mobilenet_fpn(
            backbone_name.replace('_fpn', ''), num_classes, pretrained_backbone=pretrained_backbone,
            min_size=min_size, max_size=max_size)

        if pretrained:
            print('WARNING: COCO pretrained model not available for MobileNet FPN backbones')
    else:
        raise ValueError('Invalid backbone_name: {}'.format(backbone_name))

    return model


def maskrcnn_mobilenetv2_fpn(num_classes, pretrained_backbone=False, min_size=1024, max_size=2048):
    """Builds a Mask R-CNN with MobileNetV2 FPN backbone"""
    return _maskrcnn_mobilenet_fpn('mobilenet_v2', num_classes, pretrained_backbone, min_size, max_size)


def maskrcnn_mobilenetv3_large_fpn(num_classes, pretrained_backbone=False, min_size=1024, max_size=2048):
    """Builds a Mask R-CNN with MobileNetV3 Large FPN backbone"""
    return _maskrcnn_mobilenet_fpn('mobilenet_v3_large', num_classes, pretrained_backbone, min_size, max_size)


def maskrcnn_mobilenetv3_small_fpn(num_classes, pretrained_backbone=False, min_size=1024, max_size=2048):
    """Builds a Mask R-CNN with MobileNetV3 Large FPN backbone"""
    return _maskrcnn_mobilenet_fpn('mobilenet_v3_small', num_classes, pretrained_backbone, min_size, max_size)


def _maskrcnn_mobilenet_fpn(backbone_name, num_classes, pretrained_backbone=False, min_size=1024, max_size=2048):
    """Builds a Mask R-CNN with MobileNet FPN backbone"""
    # Get MobileNet FPN backbone
    backbone = mobilenet_backbone(backbone_name, pretrained_backbone, True)

    # Make anchor generator
    anchor_sizes = ((32, 64, 128, 256, 512,),)*3
    aspect_ratios = ((0.5, 1.0, 2.0),)*len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    # Make Mask R-CNN model
    model = MaskRCNN(
        backbone, num_classes, rpn_anchor_generator=anchor_generator, min_size=min_size, max_size=max_size)

    return model
