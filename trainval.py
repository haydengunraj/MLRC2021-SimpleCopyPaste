import os
import yaml
import glob
import torch
from detection.utils import collate_fn
from detection.engine import train_one_epoch, evaluate

from model import maskrcnn_from_config
from data import get_cityscapes_dataset
from metrics import MetricManager, ScalarMetric

TORCH_HOME = 'weights'
LOSS_KEYS = ('loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg')
EXPERIMENT_DIR = 'experiments'


def save_checkpoint(output_dir, epoch, step, model, optimizer, lr_scheduler, scaler=None):
    """Saves a training checkpoint to allow training to be resumed"""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    if scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()
    torch.save(checkpoint, os.path.join(output_dir, 'checkpoint-{:04d}.pth'.format(epoch + 1)))


def get_latest_checkpoint(experiment_dir):
    """Gets the latest checkpoint for an experiment"""
    checkpoints = sorted(glob.glob(os.path.join(experiment_dir, 'checkpoints', '*.pth')))
    if len(checkpoints):
        return checkpoints[-1]
    return None


def load_config(config_file):
    """Loads a YAML configuration file"""
    if not os.path.exists(config_file):
        raise ValueError('Config file not found: {}'.format(config_file))
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def trainval_cityscapes(experiment_name, resume=False, gpus=(0,), num_workers=32):
    """Runs training and evaluation of a Mask-RCNN model on the Cityscapes dataset"""
    # Load configuration
    experiment_dir = os.path.join(EXPERIMENT_DIR, experiment_name)
    config_file = os.path.join(experiment_dir, 'config.yaml')
    config = load_config(config_file)

    # Make directories for logs and checkpoints
    log_dir = os.path.join(experiment_dir, 'logs')
    ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Set up device
    device = torch.device('cuda:{}'.format(gpus[0]))

    # Make datasets and data loaders
    print('Building datasets...', end='', flush=True)
    train_dataset = get_cityscapes_dataset(
        config['data_root'], 'train', jitter_mode=config['jitter_mode'],
        copy_paste=config['copy_paste'], image_size=(config['max_size'], config['min_size']),
        fraction=config.get('fraction', None), fraction_seed=config.get('fraction_seed', None))
    val_dataset = get_cityscapes_dataset(
        config['data_root'], 'val', image_size=(config['max_size'], config['min_size']))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=(len(gpus) > 1))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn)
    print('done', flush=True)

    # Make model
    print('Building model...', end='')
    os.environ['TORCH_HOME'] = TORCH_HOME
    model = maskrcnn_from_config(config)
    model.to(device)
    model_wo_dp = model
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
        model_wo_dp = model.module
    print('done')

    # Make optimizer and LR scheduler
    print('Building optimizer...', end='')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config['lr'], momentum=config['momentum'],
        weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config['step_size'], gamma=config['gamma'])
    print('done')

    # Make mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Load checkpoint
    init_epoch = 0
    step = 0
    if resume:
        latest_ckpt = get_latest_checkpoint(experiment_dir)
        if latest_ckpt is not None:
            state_dict = torch.load(latest_ckpt, map_location=device)
            model_wo_dp.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
            scaler.load_state_dict(state_dict['scaler'])
            init_epoch = state_dict['epoch'] + 1
            step = state_dict['step']
    elif config['checkpoint'] is not None:
        state_dict = torch.load(config['checkpoint'], map_location=device)
        model_wo_dp.load_state_dict(state_dict['model'])

    # Make metric manager
    metrics = [ScalarMetric(
        'total_loss', log_interval=config['log_interval'], scalar_key='total_loss')]
    for loss_key in LOSS_KEYS:
        metrics.append(ScalarMetric(
            loss_key, log_interval=config['log_interval'], scalar_key=loss_key))
    metrics = MetricManager(
        log_dir, metrics=metrics, purge_step=(step + 1 if step else None))

    # Main training/val loop
    print('Starting training')
    for epoch in range(init_epoch, config['epochs']):
        # Log learning rate
        metrics.writer.add_scalar(
            metrics.train_prefix + 'lr', lr_scheduler.get_last_lr(), step)

        # Train for an epoch
        step = train_one_epoch(
            model, optimizer, train_loader, device, epoch, step, metrics=metrics, scaler=scaler)
        lr_scheduler.step()
        metrics.reset()

        if (epoch + 1) == config['epochs'] or not ((epoch + 1) % config['eval_interval']):
            # Save weights and run evaluation
            save_checkpoint(ckpt_dir, epoch, step, model_wo_dp, optimizer, lr_scheduler, scaler)
            coco_evaluator = evaluate(model_wo_dp, val_loader, device=device)

            # Log eval metrics
            for iou_type, coco_eval in coco_evaluator.coco_eval.items():
                mean_ap = coco_eval.stats[0]
                iou_str = 'box_mAP' if iou_type == 'bbox' else 'mask_mAP'
                metrics.writer.add_scalar(metrics.val_prefix + iou_str, mean_ap, step)

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, help='Name of experiment directory')
    parser.add_argument('-r', '--resume', action='store_true', help='Flag to resume training from latest checkpoint')
    parser.add_argument('-g', '--gpus', type=str, default='0', help='GPU indices, separated by commas')
    parser.add_argument('-n', '--num_workers', type=int, default=32, help='Number of workers for data loading')
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(',')]
    trainval_cityscapes(args.experiment_name, resume=args.resume, gpus=gpus, num_workers=args.num_workers)
