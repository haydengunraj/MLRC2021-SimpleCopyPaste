import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.datasets import Cityscapes
from torchvision.transforms import functional as F
from detection import transforms as T

NUM_CLASSES = 9  # 8 + background
_INSTANCE_SELECTION_MODES = ('one', 'subset', 'all')


class CityscapesInstanceDataset(Cityscapes):
    """Modifies the torchvision Cityscapes dataset class to
     use the target format of torchvision detection models"""
    def __init__(self, root, split='train', transforms=None, clean=True,
                 image_size=None, fraction=None, fraction_seed=None):
        super().__init__(
            root=root, split=split, mode='fine', target_type='instance', transforms=None)
        self.transforms = transforms
        self.image_size = tuple(image_size) if image_size is not None else image_size
        if split == 'train' and clean:
            self._remove_images_without_annotations()

        # Expose only a fraction of the data
        self.fraction = fraction
        self.fraction_seed = fraction_seed
        if fraction is not None:
            # Sort before shuffling
            sorting_order = np.argsort(self.images)
            self.images = [self.images[i] for i in sorting_order]
            self.targets = [self.targets[i] for i in sorting_order]

            # Shuffle with optional fixed seed
            if fraction_seed is not None:
                np.random.seed(fraction_seed)
            order = np.arange(len(self.images), dtype=np.int32)
            np.random.shuffle(order)
            order = order[:round(fraction*len(self.images))]
            self.images = [self.images[i] for i in order]
            self.targets = [self.targets[i] for i in order]

    def _remove_images_without_annotations(self):
        """Helper to remove images that have no annotations"""
        valid_indices = []
        for i in range(len(self.images)):
            target = self._get_target(i)
            if len(target['boxes']):
                valid_indices.append(i)
        self.images = [self.images[i] for i in valid_indices]
        self.targets = [self.targets[i] for i in valid_indices]

    def _get_target(self, index):
        """Helper to convert instance ID masks to target dicts"""
        # Load and resize instance mask
        instance_mask = Image.open(self.targets[index][0])
        if self.image_size is not None and self.image_size != instance_mask.size:
            instance_mask = instance_mask.resize(self.image_size, resample=Image.NEAREST)
        instance_mask = np.asarray(instance_mask)

        # Get object instance ids
        # Here we remove non-instance classes as well as the caravan and trailer classes
        to_remove = (instance_mask < 1000) | ((instance_mask >= 29000) & (instance_mask < 31000))
        instance_mask[to_remove] = 0
        object_ids = np.unique(instance_mask)
        object_ids = object_ids[1:]

        # Make per-instance binary masks
        masks = instance_mask == object_ids[:, None, None]

        # Get bounding boxes of instances
        boxes = []
        good_indices = []  # samples where at least one instance with non-zero width exists
        for i, mask in enumerate(masks):
            y, x = np.where(mask)
            xmin = x.min()
            ymin = y.min()
            xmax = x.max()
            ymax = y.max()
            if xmin < xmax and ymin < ymax:
                good_indices.append(i)
                boxes.append([x.min(), y.min(), x.max(), y.max()])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Get areas
        if len(boxes):
            areas = (boxes[:, 3] - boxes[:, 1])*(boxes[:, 2] - boxes[:, 0])
        else:
            areas = torch.tensor([], dtype=torch.float32)

        # Instances are labelled as 1000*class + inst_number, and instance classes start at index 24
        # To map to class indices (i.e., model labels), we have to integer divide by 1000 and subtract 23.
        # Additionally, the caravan (29) and trailer (30) classes are not considered, so
        # for classes above these we also have to subtract 2.
        labels = object_ids[good_indices]//1000
        labels[labels > 30] -= 2
        labels -= 23

        # Create target dict
        target = {
            'image_id': torch.tensor([index]),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'boxes': boxes,
            'masks': torch.as_tensor(masks[good_indices], dtype=torch.uint8),
            'area': areas,
            'iscrowd': torch.zeros((len(good_indices),), dtype=torch.int64)
        }

        return target

    def __getitem__(self, index):
        # Load image and create target
        image = Image.open(self.images[index]).convert('RGB')
        if self.image_size is not None and self.image_size != image.size:
            image = image.resize(self.image_size, resample=Image.BILINEAR)
        target = self._get_target(index)

        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class CityscapesCopyPasteInstanceDataset(CityscapesInstanceDataset):
    """Modifies the CityscapesInstanceDataset to add Copy-Paste Augmentation"""
    def __init__(self, root, split='train', transforms=None, clean=True, image_size=None,
                 fraction=None, fraction_seed=None, selection_mode='subset',
                 occluded_obj_thresh=20, p=0.5):
        super().__init__(
            root=root, split=split, transforms=transforms, clean=clean,
            image_size=image_size, fraction=fraction, fraction_seed=fraction_seed)
        if selection_mode not in _INSTANCE_SELECTION_MODES:
            raise ValueError('Invalid selection_mode: {}. Must be one of: {}'.format(
                selection_mode, _INSTANCE_SELECTION_MODES))
        self.selection_mode = selection_mode
        self.occluded_obj_thresh = occluded_obj_thresh
        self.p = p

    def __getitem__(self, index):
        # Load image and target
        image, target = super().__getitem__(index)

        if self.split != 'train' or torch.rand(1) > self.p:
            return image, target

        # Get random second image and target
        img_indices = np.arange(len(self.images), dtype=np.int32)
        img_indices[index] = index - 1 if index else index + 1  # ensure same image not selected
        src_image, src_target = super().__getitem__(np.random.choice(img_indices))

        # Paste objects from src_image into image
        image, target = copy_paste_augmentation(
            image, target, src_image, src_target, selection_mode=self.selection_mode,
            occluded_obj_thresh=self.occluded_obj_thresh)

        return image, target


def copy_paste_augmentation(image, target, src_image, src_target, selection_mode='subset', occluded_obj_thresh=20):
    """Performs Copy-Paste augmentation for a pair of images and
    targets. Instances are pasted from src_image into image."""
    # Select object indices from second image
    num_objs = len(src_target['labels'])
    if not num_objs:
        return image, target
    obj_indices = np.arange(num_objs, dtype=np.int32)  # all
    if selection_mode == _INSTANCE_SELECTION_MODES[0]:  # one
        obj_indices = np.atleast_1d(np.random.choice(obj_indices))
    elif selection_mode == _INSTANCE_SELECTION_MODES[1]:  # random subset
        num_inst = np.random.randint(1, num_objs + 1)
        obj_indices = np.random.choice(obj_indices, num_inst, replace=False)

    # Get overall mask of instances
    src_mask = torch.any(src_target['masks'][obj_indices], dim=0, keepdim=True)

    # Remove overall mask from masks in original image
    new_masks = target['masks'] - np.logical_and(target['masks'], src_mask)

    # Update target to fix occluded masks
    target = update_occluded_masks(target, new_masks, occluded_obj_thresh=occluded_obj_thresh)

    # Add pasted instances to target
    for key in ('labels', 'boxes', 'masks', 'area', 'iscrowd'):
        target[key] = torch.cat((target[key], src_target[key][obj_indices]), dim=0)

    # Paste instances into original image
    src_mask = src_mask[0].type(torch.bool)
    image[:, src_mask] = src_image[:, src_mask]

    return image, target


class RandomScaleJitter(nn.Module):
    """Implements random scale jitter"""
    def __init__(self, scale_range=(0.8, 1.2), occluded_obj_thresh=20, p=0.5):
        super().__init__()
        self.fill = [127.5, 127.5, 127.5]
        self.scale_range = scale_range
        self.occluded_obj_thresh = occluded_obj_thresh
        if scale_range[0] > scale_range[1]:
            raise ValueError('Invalid scale range: {}'.format(scale_range))
        self.p = p

    @torch.jit.unused
    def _get_fill_value(self, is_pil):
        # type: (bool) -> int
        return tuple(int(x) for x in self.fill) if is_pil else 0

    def forward(self, image, target=None):
        if torch.rand(1) > self.p:
            return image, target

        if torch.jit.is_scripting():
            fill = 0
        else:
            fill = self._get_fill_value(F._is_pil_image(image))

        # Get new image size from scale
        width, height = F.get_image_size(image)
        scale = self.scale_range[0] + torch.rand(1)*(self.scale_range[1] - self.scale_range[0])
        new_width = int(width*scale)
        new_height = int(height*scale)

        if scale < 1:
            # Resize and pad image
            right_pad = width - new_width
            bottom_pad = height - new_height
            image = F.resize(image, [new_height, new_width])
            image = F.pad(image, [0, 0, right_pad, bottom_pad], fill=fill)

            if target is not None:
                # Resize and pad masks w/ bbox updates
                for i, mask in enumerate(target['masks']):
                    mask = mask.unsqueeze(0)
                    mask = F.resize(mask, [new_height, new_width], interpolation=F.InterpolationMode.NEAREST)
                    mask = F.pad(mask, [0, 0, right_pad, bottom_pad], fill=0)
                    target['masks'][i] = mask
                    target['boxes'][i] = target['boxes'][i]*scale
        elif scale > 1:
            # Resize and randomly crop image
            top = int(torch.rand(1)*(new_height - height))
            left = int(torch.rand(1)*(new_width - width))
            image = F.resize(image, [new_height, new_width])
            image = F.crop(image, top, left, height, width)

            if target is not None:
                # Resize and crop masks
                new_masks = []
                for mask in target['masks']:
                    mask = mask.unsqueeze(0)
                    mask = F.resize(mask, [new_height, new_width], interpolation=F.InterpolationMode.NEAREST)
                    mask = F.crop(mask, top, left, height, width)
                    new_masks.append(mask)
                new_masks = torch.cat(new_masks, dim=0)

                # Update target
                target = update_occluded_masks(target, new_masks, occluded_obj_thresh=self.occluded_obj_thresh)

        return image, target


def update_occluded_masks(target, new_masks, occluded_obj_thresh):
    """Edits targets to remove occluded masks and update boxes and areas"""
    # Check for masks to be kept
    new_mask_sizes = torch.sum(new_masks, dim=(1, 2))
    good_indices = torch.nonzero(new_mask_sizes > occluded_obj_thresh, as_tuple=True)[0]

    # Update bounding boxes and areas, checking
    # for degenerate bounding boxes as well
    valid_box_indices = []
    for i in good_indices:
        y, x = torch.nonzero(new_masks[i], as_tuple=True)
        xmin = x.min()
        ymin = y.min()
        xmax = x.max()
        ymax = y.max()
        if xmin < xmax and ymin < ymax:
            valid_box_indices.append(i)
            target['boxes'][i, 0] = x.min()
            target['boxes'][i, 1] = y.min()
            target['boxes'][i, 2] = x.max()
            target['boxes'][i, 3] = y.max()
            target['area'][i] = (target['boxes'][i, 3] - target['boxes'][i, 1])*(
                    target['boxes'][i, 2] - target['boxes'][i, 0])
    target['masks'] = new_masks

    # Remove occluded objects from target
    valid_box_indices = torch.as_tensor(valid_box_indices, dtype=torch.int64)
    for key in ('labels', 'boxes', 'masks', 'area', 'iscrowd'):
        target[key] = target[key][valid_box_indices]

    return target


def get_transform(is_training, jitter_mode='standard'):
    """Creates transforms for the Cityscapes dataset"""
    if is_training:
        if jitter_mode == 'standard':
            scale_range = [0.8, 1.25]
        elif jitter_mode == 'large':
            scale_range = [0.1, 2]
        else:
            raise ValueError('Invalid scale jitter mode: {}'.format(jitter_mode))
        transforms = [
            T.RandomHorizontalFlip(p=0.5),
            RandomScaleJitter(scale_range=scale_range, occluded_obj_thresh=20, p=0.5)
        ]
    else:
        transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def get_cityscapes_dataset(root, split, jitter_mode='standard', copy_paste=True,
                           image_size=None, fraction=None, fraction_seed=None):
    """Helper to create Cityscapes train/val datasets"""
    if split == 'train':
        train_tform = get_transform(True, jitter_mode=jitter_mode)
        if copy_paste:
            dataset = CityscapesCopyPasteInstanceDataset(
                root=root, split=split, transforms=train_tform, image_size=image_size,
                fraction=fraction, fraction_seed=fraction_seed)
        else:
            dataset = CityscapesInstanceDataset(
                root=root, split=split, transforms=train_tform, image_size=image_size,
                fraction=fraction, fraction_seed=fraction_seed)
    elif split in ('val', 'test'):
        dataset = CityscapesInstanceDataset(
            root=root, split=split, transforms=get_transform(False), image_size=image_size)
    else:
        raise ValueError('Invalid split: {}'.format(split))

    return dataset
