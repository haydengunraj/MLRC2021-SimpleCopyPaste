import os
import cv2
import torch
import numpy as np

from model import mask_rcnn
from data import get_cityscapes_dataset, NUM_CLASSES


def _format_instance_dict(instance_dict):
    """Convert Torch detections/annotations to numpy"""
    # Convert to numpy arrays
    instance_dict = {key: val.cpu().numpy() for key, val in instance_dict.items()}

    # Add label offsets and convert masks to boolean arrays
    instance_dict['labels'] += 23
    instance_dict['masks'] = instance_dict['masks'] >= 0.5
    instance_dict['boxes'] = instance_dict['boxes'].astype(np.int32)

    return instance_dict


def outline_mask(image, mask, colour=(255, 255, 255)):
    """Draws an outline around a mask"""
    if mask.dtype == bool:
        mask = mask.astype(np.uint8)
    contours = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(image, contours, -1, colour, thickness=2)


def overlay_masks(image, instance_dict, classes, alpha=0.4):
    """Overlays semi-transparent instance masks on an image"""
    if len(instance_dict['labels']):
        # Make RGB mask for overlay
        masks = instance_dict['masks']
        labels = instance_dict['labels']
        if masks.ndim == 4:
            masks = masks.squeeze(axis=1)
        rgb_mask = np.zeros(masks[0].shape + (3,), dtype=np.float32)
        for cls_idx, mask in zip(labels, masks):
            # Get class color
            cls = classes[cls_idx]
            colour = cls.color

            # Add colour to RGB mask
            rgb_mask[mask, :] = colour

        # Alpha blend RGB mask with image
        full_mask = np.any(masks, axis=0)
        alpha_mask = alpha*full_mask.astype(np.float32)[..., None]
        blended = image.astype(np.float32)*(1 - alpha_mask) + rgb_mask*alpha_mask
        blended = blended.astype(np.uint8)

        # Add white outlines to instances
        for mask in masks:
            outline_mask(blended, mask)

        return blended
    else:
        return image


def overlay_bboxes(image, instance_dict, classes, colour=(255, 255, 255),
                   font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1):
    """Overlays bounding boxes on an image"""
    if len(instance_dict['labels']):
        for i, (cls_idx, box) in enumerate(zip(instance_dict['labels'], instance_dict['boxes'])):
            # Get class name
            cls = classes[cls_idx]
            name = cls.name

            # Draw bbox
            up_left, low_right = tuple(box[:2]), tuple(box[2:])
            cv2.rectangle(image, up_left, low_right, colour, 1)

            # Add bbox label
            box_label = name
            if 'scores' in instance_dict:
                box_label += '-{:.2f}'.format(instance_dict['scores'][i])

            cv2.putText(
                image, box_label, (up_left[0], up_left[1] - 2*font_thickness),
                font, font_scale, colour, font_thickness)

    return image


def visualize_cityscapes(cityscapes_root, checkpoint, output_dir, include_boxes=True,
                         num_vis=100, score_thresh=0.5, device='cuda:0'):
    """Makes visualizations of a trained Mask-RCNN model's predictions"""
    # Setup device
    device = torch.device(device)

    # Make dataset and data loader
    dataset = get_cityscapes_dataset(cityscapes_root, 'val')

    # Make model
    model = mask_rcnn(NUM_CLASSES, pretrained=False, pretrained_backbone=False)
    model.to(device)

    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict['model'])
    model.eval()

    # Select indices to visualize
    indices = np.random.choice(np.arange(len(dataset)), num_vis, replace=False)

    with torch.no_grad():
        for idx in indices:
            # Load image and target
            image, target = dataset[idx]
            target = _format_instance_dict(target)

            # Run model
            detections = model(image.unsqueeze(0).to(device))[0]
            detections = _format_instance_dict(detections)
            valid_idx = detections['scores'] >= score_thresh
            for key in ('boxes', 'masks', 'labels', 'scores'):
                detections[key] = detections[key][valid_idx]

            # Make ground-truth image
            image = np.moveaxis(image.cpu().numpy()*255, 0, -1).astype(np.uint8)
            image_gt = overlay_masks(image, target, dataset.classes)
            if include_boxes:
                image_gt = overlay_bboxes(image_gt, target, dataset.classes)

            # Make prediction image
            image_pred = overlay_masks(image, detections, dataset.classes)
            if include_boxes:
                image_pred = overlay_bboxes(image_pred, detections, dataset.classes)

            # Concatenate ground-truth and predictions
            spacer = 255*np.ones((image.shape[0], 25, 3), dtype=np.uint8)
            vis = np.concatenate((image_gt, spacer, image_pred), axis=1)

            # Save image
            out_file = os.path.join(output_dir, 'vis-{:04d}.png'.format(idx))
            cv2.imwrite(out_file, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('-rt', '--root', type=str, default='cityscapes', help='Path to Cityscapes data directory')
    parser.add_argument('-op', '--output', type=str, default='visualizations', help='Output directory')
    parser.add_argument('-nv', '--num_visualizations', type=int, default=50,
                        help='Number of visualizations to generate')
    parser.add_argument('-nb', '--no_bboxes', action='store_true', help='Flag to prevent overlay of bounding boxes')
    parser.add_argument('-gd', '--gpu_device', type=int, default=0, help='GPU index')
    args = parser.parse_args()

    # Run training
    visualize_cityscapes(
        args.root,
        args.checkpoint,
        args.output,
        include_boxes=(not args.no_bboxes),
        num_vis=args.num_visualizations,
        device='cuda:{}'.format(args.gpu_device)
    )
