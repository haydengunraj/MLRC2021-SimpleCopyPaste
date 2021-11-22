import torch
import numpy as np
from PIL import Image
from torchvision.datasets import Cityscapes
from detection import transforms as T

NUM_CLASSES = 11


class CityscapesInstanceDataset(Cityscapes):
    """Modifies the built-in Cityscapes dataset class to
     use the target format of torchvision detection models"""
    def __init__(self, root, split='train', transforms=None):
        super().__init__(
            root=root, split=split, mode='fine', target_type='instance', transforms=None)
        self.transforms = transforms
        if split == 'train':
            self._remove_images_without_annotations()

    def _remove_images_without_annotations(self):
        """Helper to remove images that have no annotations"""
        valid_indices = []
        for i in range(len(self)):
            target = self._get_target(i)
            if len(target['boxes']):
                valid_indices.append(i)
        self.images = [self.images[i] for i in valid_indices]
        self.targets = [self.targets[i] for i in valid_indices]

    def _get_target(self, index):
        """Helper to convert instance ID masks to target dicts"""
        # Get object instance ids
        instance_mask = Image.open(self.targets[index][0])
        instance_mask = np.asarray(instance_mask)
        instance_mask[instance_mask < 1000] = 0
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

        # Create target dict
        target = {
            'image_id': torch.tensor([index]),
            'labels': torch.as_tensor(object_ids[good_indices]//1000 - 23, dtype=torch.int64),
            'boxes': boxes,
            'masks': torch.as_tensor(masks[good_indices], dtype=torch.uint8),
            'area': areas,
            'iscrowd': torch.zeros((len(good_indices),), dtype=torch.int64)
        }

        return target

    def __getitem__(self, index):
        # Load image and create target
        image = Image.open(self.images[index]).convert('RGB')
        target = self._get_target(index)

        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class CityscapesCopyPasteInstanceDataset(CityscapesInstanceDataset):
    def __init__(self, root, split='train', transforms=None):
        super().__init__(
            root=root, split=split, transforms=transforms)
        self.transforms = transforms


class CopyPaste:
    def __init__(self, occluded_obj_threshold=300, box_update_threshold=10, p=0.5):
        pass

    def __call__(self, source_image, dest_image, source_mask, dest_mask):
        raise NotImplementedError


def get_cityscapes_dataset(root):
    train_dataset = CityscapesInstanceDataset(
        root=root, split='train', transforms=get_transform(True))
    val_dataset = CityscapesInstanceDataset(
        root=root, split='val', transforms=get_transform(False))

    return train_dataset, val_dataset


def get_transform(is_training):
    transforms = [T.ToTensor()]
    if is_training:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# if __name__ == '__main__':
#     root = 'D:\\Datasets\\Cityscapes'
#     import cv2
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     dset = CityscapesInstanceDataset(root, split='val')
#
#     idx = 5
#     im, tar = dset[idx]
#
#     im = np.asarray(im).astype(np.uint8)
#     classes = tar['labels'].numpy()
#     index_map = {cls: i for i, cls in enumerate(np.unique(classes))}
#     boxes = tar['boxes'].numpy().astype(np.int32)
#
#     colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
#
#     for cls, box in zip(classes, boxes):
#         cv2.rectangle(im, tuple(box[:2]), tuple(box[2:]), colours[index_map[cls]], 2)
#
#
#     # tar = np.asarray(tar)
#     #
#     # print(np.unique(tar))
#     #
#     #
#     fig, ax = plt.subplots()
#     ax.imshow(im)
#     plt.show()
#
#     # print(tar.min(), tar.max(), tar.shape, tar.dtype)
