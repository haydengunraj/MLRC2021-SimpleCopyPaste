import os
import torch
from detection.utils import collate_fn
from detection.engine import train_one_epoch, evaluate

from model import mask_rcnn
from data import get_cityscapes_dataset, NUM_CLASSES
from metrics import MetricManager, ScalarMetric

TORCH_HOME = 'weights'
LOSS_KEYS = ('loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg')


def save_checkpoint(output_dir, epoch, model, optimizer, lr_scheduler, scaler=None):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
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
        epochs=10,
        eval_interval=5,
        batch_size=2,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
        step_size=50,
        gamma=0.1,
        num_workers=4,
        device='cuda:0'
):
    # Setup device
    device = torch.device(device)

    # Make datasets and data loaders
    print('Building datasets...', end='', flush=True)
    train_dataset, val_dataset = get_cityscapes_dataset(cityscapes_root)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn)
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

    # Make metric manager
    metrics = [ScalarMetric('total_loss', log_interval=50, scalar_key='total_loss')]
    for loss_key in LOSS_KEYS:
        metrics.append(ScalarMetric(loss_key, log_interval=50, scalar_key=loss_key))
    metrics = MetricManager(
        os.path.join(output_dir, 'logs'),
        metrics=metrics
    )

    # Main training/val loop
    print('Starting training')
    scaler = torch.cuda.amp.GradScaler()
    step = 0
    for epoch in range(epochs):
        # Train for an epoch
        train_one_epoch(model, optimizer, train_loader, device, epoch, step, metrics=metrics, scaler=scaler)
        lr_scheduler.step()
        metrics.reset()

        if not ((epoch + 1) % eval_interval):
            # Run eval process
            coco_evaluator = evaluate(model, val_loader, device=device)
            save_checkpoint(output_dir, epoch, model, optimizer, lr_scheduler, scaler)

            # Log eval metrics
            for iou_type, coco_eval in coco_evaluator.coco_eval.items():
                mean_ap = coco_eval.stats[0]
                iou_str = 'box_mAP' if iou_type == 'bbox' else 'mask_mAP'
                metrics.writer.add_scalar(metrics.val_prefix + iou_str, mean_ap, step)

    return model


if __name__ == '__main__':
    root = 'cityscapes'
    out_dir = 'checkpoints'
    epochs = 200
    eval_interval = 10
    batch_size = 16
    lr = 0.02
    momentum = 0.9
    weight_decay = 0.0001

    trainval_cityscapes(
        root,
        out_dir,
        epochs=epochs,
        eval_interval=eval_interval,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
