import os
import torch
from detection.utils import collate_fn
from detection.engine import evaluate

from model import maskrcnn_from_config
from data import get_cityscapes_dataset
from trainval import load_config


def evaluate_cityscapes(cityscapes_root, checkpoint, config, num_workers=4, device='cuda:0'):
    """Runs evaluation of a trained Mask-RCNN model on the Cityscapes dataset"""
    # Setup device
    device = torch.device(device)

    # Make dataset and data loader
    val_dataset = get_cityscapes_dataset(cityscapes_root, 'val', image_size=(config['max_size'], config['min_size']))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # Make model
    model = maskrcnn_from_config(config, override_pretraining=True)
    model.to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict['model'])

    # Run evaluation
    evaluate(model, val_loader, device=device)


if __name__ == '__main__':
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('-f', '--config_file', type=str, help='Configuration file used to build the model')
    parser.add_argument('-r', '--root', type=str, default='cityscapes', help='Path to Cityscapes data directory')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('-n', '--num_workers', type=int, default=1, help='Number of workers for data loading')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config_file)

    # Run training
    evaluate_cityscapes(
        args.root,
        args.checkpoint,
        config,
        num_workers=args.num_workers,
        device='cuda:{}'.format(args.gpu)
    )
