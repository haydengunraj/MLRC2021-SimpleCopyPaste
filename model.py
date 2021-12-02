from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor, maskrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import misc as misc_nn_ops
import torch

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
    elif backbone_name == 'mobilenetv2_fpn':
        # Build Mask R-CNN MobileNetV2 FPN
        model = maskrcnn_mobilenetv2_fpn(
            num_classes, pretrained_backbone=pretrained_backbone, min_size=min_size, max_size=max_size)

        if pretrained:
            print('WARNING: COCO pretrained model not available for MobileNetV2 FPN backbone')
    elif backbone_name == 'efficientnetb7_fpn':
        # Build Mask R-CNN EfficientNet B7 FPN
        model = maskrcnn_efficientnetb7_fpn(
            num_classes, pretrained_backbone=pretrained_backbone, 
            min_size=min_size, max_size=max_size)
    else:
        raise ValueError('Invalid backbone_name: {}'.format(backbone_name))

    return model


def maskrcnn_mobilenetv2_fpn(num_classes, pretrained_backbone=False, min_size=1024, max_size=2048):
    """Builds a Mask R-CNN with MobileNetV2 FPN backbone"""
    # MobileNetV2 FPN
    backbone = mobilenet_backbone(
        'mobilenet_v2',
        pretrained_backbone,
        True,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
        trainable_layers=2,
        returned_layers=None,
        extra_blocks=None
    )
    # Make anchor generator
    anchor_generator = AnchorGenerator(
        sizes=(32, 64, 128),
        aspect_ratios=(0.5, 1.0, 2.0))

    model = MaskRCNN(
        backbone, num_classes, rpn_anchor_generator=anchor_generator, min_size=min_size, max_size=max_size)
    return model


def maskrcnn_efficientnetb7_fpn(num_classes, pretrained_backbone=False, 
                                min_size=1024, max_size=2048):
    """Builds a Mask R-CNN with EfficientNet B7 FPN backbone."""

    # EfficientNet B7 FPN backbone
    backbone = torch.hub.load(
        'AdeelH/pytorch-fpn',
        'make_fpn_efficientnet',
        force_reload=True,
        name='efficientnet_b7',
        fpn_type='fpn',
        pretrained=pretrained_backbone,
        num_classes=num_classes,
        fpn_channels=256,
        in_channels=3,
        out_size=(224, 224)
    )

    backbone.out_channels = 256
    model = MaskRCNN(backbone=backbone, num_classes=num_classes, 
                    min_size=min_size, max_size=max_size)

    return model
