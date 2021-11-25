import os
import torch
from detection.utils import collate_fn
from detection.engine import train_one_epoch, evaluate

from model import mask_rcnn
from data import get_cityscapes_dataset, NUM_CLASSES
from metrics import MetricManager, ScalarMetric

TORCH_HOME = 'weights'
LOSS_KEYS = ('loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg')
EXPERIMENT_DIR = 'experiments'


def save_checkpoint(output_dir, epoch, step, model, optimizer, lr_scheduler, scaler=None):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    if scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()
    torch.save(checkpoint, os.path.join(output_dir, 'checkpoint-{:04d}.pth'.format(epoch)))


def trainval_cityscapes(
        cityscapes_root,
        output_dir,
        checkpoint=None,
        epochs=10,
        eval_interval=5,
        batch_size=2,
        copy_paste=True,
        jitter_mode='standard',
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
        step_size=50,
        gamma=0.1,
        num_workers=4,
        device='cuda:0'
):
    """Runs training and evaluation of a Mask-RCNN model on the Cityscapes dataset"""
    # Setup device
    device = torch.device(device)

    # Make datasets and data loaders
    print('Building datasets...', end='', flush=True)
    train_dataset = get_cityscapes_dataset(
        cityscapes_root, 'train', jitter_mode=jitter_mode, copy_paste=copy_paste)
    val_dataset = get_cityscapes_dataset(cityscapes_root, 'val')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    print('done', flush=True)

    # Make model
    print('Building model...', end='')
    os.environ['TORCH_HOME'] = TORCH_HOME
    model = mask_rcnn(NUM_CLASSES, pretrained=False, pretrained_backbone=True)
    model.to(device)
    print('done')

    # Make optimizer and LR scheduler
    print('Building optimizer...', end='')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    print('done')

    # Make mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Load checkpoint
    if checkpoint is None:
        init_epoch = 0
        step = 0
    else:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        scaler.load_state_dict(state_dict['scaler'])
        init_epoch = state_dict['epoch']
        step = state_dict['step']

    # Make metric manager
    metrics = [ScalarMetric('total_loss', log_interval=50, scalar_key='total_loss')]
    for loss_key in LOSS_KEYS:
        metrics.append(ScalarMetric(loss_key, log_interval=50, scalar_key=loss_key))
    metrics = MetricManager(
        os.path.join(output_dir, 'logs'),
        metrics=metrics, purge_step=(step if step else None)
    )

    # Main training/val loop
    print('Starting training')
    for epoch in range(init_epoch, epochs):
        # Train for an epoch
        step = train_one_epoch(model, optimizer, train_loader, device, epoch, step, metrics=metrics, scaler=scaler)
        lr_scheduler.step()
        metrics.reset()

        if not ((epoch + 1) % eval_interval):
            # Save weights and run evaluation
            save_checkpoint(output_dir, epoch, step, model, optimizer, lr_scheduler, scaler)
            coco_evaluator = evaluate(model, val_loader, device=device)

            # Log eval metrics
            for iou_type, coco_eval in coco_evaluator.coco_eval.items():
                mean_ap = coco_eval.stats[0]
                iou_str = 'box_mAP' if iou_type == 'bbox' else 'mask_mAP'
                metrics.writer.add_scalar(metrics.val_prefix + iou_str, mean_ap, step)

    return model


if __name__ == '__main__':
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='Name for the training run')
    parser.add_argument('-rt', '--root', type=str, default='cityscapes', help='Path to Cityscapes data directory')
    parser.add_argument('-ep', '--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('-ck', '--checkpoint', type=str, default='', help='Checkpoint file')
    parser.add_argument('-ei', '--eval_interval', type=int, default=10, help='Interval between evaluations, in epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('-cp', '--copy_paste', action='store_true', help='Flag to enable Copy-Paste augmentation')
    parser.add_argument('-ji', '--jitter_mode', type=str, default='standard',
                        choices=('standard', 'large'), help='Scale jitter mode')
    parser.add_argument('-lr', '--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('-mo', '--momentum', type=float, default=0.9, help='Optimizer momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0001, help='Optimizer weight decay')
    parser.add_argument('-ss', '--step_size', type=int, default=100,
                        help='Number of epochs between step learning rate decay')
    parser.add_argument('-ga', '--gamma', type=float, default=0.1, help='Step learning rate decay constant')
    parser.add_argument('-gd', '--gpu_device', type=int, default=0, help='GPU index')
    parser.add_argument('-nw', '--num_workers', type=int, default=24, help='Number of dataloader workers')
    args = parser.parse_args()

    # Make output directory, device ID, and checkpoint
    output_dir = os.path.join(EXPERIMENT_DIR, args.name)
    os.makedirs(output_dir)
    device = 'cuda:{}'.format(args.gpu_device)
    checkpoint = args.checkpoint if args.checkpoint else None

    # Dump parameters to output directory
    with open(os.path.join(output_dir, 'run_settings.json'), 'w') as f:
        json.dump(vars(args), f)

    # Run training
    trainval_cityscapes(
        args.root,
        output_dir,
        checkpoint=checkpoint,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        copy_paste=args.copy_paste,
        jitter_mode=args.jitter_mode,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        num_workers=args.num_workers,
        device=device
    )
