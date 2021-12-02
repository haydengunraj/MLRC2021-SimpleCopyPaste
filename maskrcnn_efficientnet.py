# %%
import torch
from torch import nn
import torchvision.models as models
from torchvision.models import resnet
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.ops.feature_pyramid_network as fpn
# %%
efficientnet_b7 = models.efficientnet_b7()
print(efficientnet_b7)
# %%
model = MaskRCNN(backbone=efficientnet_b7, num_classes=91)
print(model)
# %%
resnet_backbone = resnet_fpn_backbone('resnet50', False)
print(resnet_backbone)

# %%
print(resnet_backbone.inplanes)
# %%
dir(resnet_backbone)
    
# %%
dir(efficientnet_b7)
# %%
backbone = efficientnet_b7
backbone.inplanes
# model = BackboneWithFPN(?)
# %%
backbone = resnet.__dict__['resnet50']()
print(backbone.inplanes)
# %%
mbnet_backbone =  mobilenet_backbone(
        'mobilenet_v2',
        False,
        True,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
        trainable_layers=2,
        returned_layers=None,
        extra_blocks=None
    )
print(mbnet_backbone)
# %%
print(mbnet_backbone.inplanes)

# %%
efficientnet_b7 = models.efficientnet_b7()
anchor_generator = AnchorGenerator(
        sizes=(32, 64, 128),
        aspect_ratios=(0.5, 1.0, 2.0))
# %%
bb_with_fpn = BackboneWithFPN(efficientnet_b7)

# %%
efficientnet_backbone = torch.hub.load(
	'AdeelH/pytorch-fpn',
	'make_fpn_efficientnet',
    force_reload=True,
	name='efficientnet_b7',
	fpn_type='fpn',
	num_classes=9,
	fpn_channels=256,
	in_channels=3,
	out_size=(224, 224)
)
print(backbone)
# %%
anchor_generator = AnchorGenerator(
        sizes=(32, 64, 128),
        aspect_ratios=(0.5, 1.0, 2.0))
# %%
efficientnet_backbone.out_channels = 256
eff_model = MaskRCNN(backbone=efficientnet_backbone, num_classes=9, min_size=1024, max_size=2048)
# %%
print(eff_model)
# %%
eff_model_anchor = MaskRCNN(backbone=efficientnet_backbone, num_classes=9, min_size=1024, max_size=2048, rpn_anchor_generator=anchor_generator)
print(eff_model_anchor)
# %%
model.train()
# %%
print(model)
# %%
print(mbnet_backbone)

# %%
mbnet_model = MaskRCNN(mbnet_backbone, num_classes=9, min_size=1024, max_size=2048, rpn_anchor_generator=anchor_generator)
# %%
print(eff_model_anchor)

# %%
print(mbnet_model)
# %%
print(eff_model_anchor)
# %%
print(eff_model)
# %%
backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained, norm_layer=norm_layer).features

# Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
# The first and last blocks are always included because they are the C0 (conv1) and Cn.
stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
num_stages = len(stage_indices)

# find the index of the layer from which we wont freeze
assert 0 <= trainable_layers <= num_stages
freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

for b in backbone[:freeze_before]:
for parameter in b.parameters():
        parameter.requires_grad_(False)

out_channels = 256
if fpn:
if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

if returned_layers is None:
        returned_layers = [num_stages - 2, num_stages - 1]
assert min(returned_layers) >= 0 and max(returned_layers) < num_stages
return_layers = {f'{stage_indices[k]}': str(v) for v, k in enumerate(returned_layers)}

in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
