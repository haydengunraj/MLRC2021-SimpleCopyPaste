import os
import cv2
import numpy as np
from copy import deepcopy

from data import CityscapesInstanceDataset, get_transform, copy_paste_augmentation
from visualize_detections import overlay_masks, overlay_bboxes, format_instance_dict


def visualize_augmentations(cityscapes_root, output_dir, jitter='standard', include_boxes=True, num_vis=100):
    """Makes visualizations of Copy-Paste augmentation"""
    # Make datasets
    tform = get_transform(True, jitter)
    tform.transforms[1].p = 1  # force scale jitter
    dataset_aug = CityscapesInstanceDataset(
        root=cityscapes_root, split='train', transforms=tform, clean=False)
    dataset_van = CityscapesInstanceDataset(
        root=cityscapes_root, split='train', transforms=get_transform(False), clean=False)

    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    # Select indices to visualize
    indices = np.random.choice(np.arange(len(dataset_aug)), 2*num_vis, replace=False)

    for i in range(num_vis):
        # Load original images and targets
        image1_orig, target1_orig = dataset_van[indices[i]]
        image2_orig, target2_orig = dataset_van[indices[i + 1]]

        # Load augmented images and targets
        image1, target1 = dataset_aug[indices[i]]
        image2, target2 = dataset_aug[indices[i + 1]]

        # Perform copy paste
        cp_image_12, cp_target_12 = copy_paste_augmentation(
            deepcopy(image2), deepcopy(target2), image1, target1, selection_mode='subset', occluded_obj_thresh=20)
        cp_image_21, cp_target_21 = copy_paste_augmentation(
            deepcopy(image1), deepcopy(target1), image2, target2, selection_mode='subset', occluded_obj_thresh=20)

        # Collect images
        images = [
            (image1_orig, target1_orig, 'image1.png'),
            (image1, target1, 'image1-aug.png'),
            (image2_orig, target2_orig, 'image2.png'),
            (image2, target2, 'image2-aug.png'),
            (cp_image_12, cp_target_12, 'copy-paste-1to2.png'),
            (cp_image_21, cp_target_21, 'copy-paste-2to1.png'),
        ]

        img_out_dir = os.path.join(output_dir, 'vis-{:04d}'.format(i))
        os.makedirs(img_out_dir, exist_ok=True)
        for image, target, fname in images:
            # Prepare target
            target = format_instance_dict(target)

            # Make overlay
            image = np.moveaxis(image.cpu().numpy()*255, 0, -1).astype(np.uint8)
            image = overlay_masks(image, target, dataset_van.classes)
            if include_boxes:
                image = overlay_bboxes(image, target, dataset_van.classes)

            # Save image
            out_file = os.path.join(img_out_dir, fname)
            cv2.imwrite(out_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, default='cityscapes', help='Path to Cityscapes data directory')
    parser.add_argument('-o', '--output', type=str, default='aug_visualizations', help='Output directory')
    parser.add_argument('-j', '--jitter_mode', type=str, default='standard', help='Jitter mode')
    parser.add_argument('-v', '--num_visualizations', type=int, default=50,
                        help='Number of visualizations to generate')
    parser.add_argument('-b', '--no_bboxes', action='store_true', help='Flag to prevent overlay of bounding boxes')
    args = parser.parse_args()

    # Run training
    visualize_augmentations(
        args.root,
        args.output,
        jitter=args.jitter_mode,
        include_boxes=(not args.no_bboxes),
        num_vis=args.num_visualizations
    )
