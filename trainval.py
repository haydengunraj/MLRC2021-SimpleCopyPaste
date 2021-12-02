import os
import yaml
import glob
import torch
from torch import nn
from detection.utils import collate_fn
from detection.engine import train_one_epoch, evaluate
import wandb
from model import maskrcnn_from_config
from data import get_cityscapes_dataset
from metrics import MetricManager, ScalarMetric

import detection.utils as utils

TORCH_HOME = 'weights'
LOSS_KEYS = ('loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg')
CONFIG_DIR = 'configurations'
EXPERIMENT_DIR = 'experiments'


def save_checkpoint(output_dir, epoch, step, model, optimizer, lr_scheduler, scaler=None, enable_wandb=False):
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

    ckpt_path = os.path.join(output_dir, 'checkpoint-{:04d}.pth'.format(epoch + 1))
    torch.save(checkpoint, ckpt_path)

    if enable_wandb:
        wandb.save(ckpt_path)


def get_latest_checkpoint(experiment_dir):
    """Gets the latest checkpoint for an experiment"""
    checkpoints = sorted(glob.glob(os.path.join(experiment_dir, 'checkpoints', '*.pth')))
    if len(checkpoints):
        return checkpoints[-1]
    return None


def trainval_cityscapes(args):
    """Runs training and evaluation of a Mask-RCNN model on the Cityscapes dataset"""
    # Setup params and Distributed mode (if needed)
    experiment_name = args.experiment_name
    resume=args.resume
    utils.init_distributed_mode(args)
    print(args)

    # Load configuration
    experiment_dir = os.path.join(EXPERIMENT_DIR, experiment_name)
    config_dir = os.path.join(CONFIG_DIR, experiment_name)
    config_file = os.path.join(config_dir, 'config.yaml')
    if not os.path.exists(config_file):
        raise ValueError('Config file not found: {}'.format(config_file))
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Setup wandb (if needed)
    if args.distributed and args.gpu != 0:
        # Only log to wandb from GPU 0 process
        args.enable_wandb = False

    if args.enable_wandb:
        wandb.init(project=args.experiment_name, entity="syde671-copy-paste", 
                    config=config)
        wandb.tensorboard.patch(pytorch=True, tensorboardX=True)


    # Make directories for logs and checkpoints
    log_dir = os.path.join(experiment_dir, 'logs')
    ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda')

    # Make datasets and data loaders
    print('Building datasets...', end='', flush=True)
    train_dataset = get_cityscapes_dataset(
        config['data_root'], 'train', jitter_mode=config['jitter_mode'],
        copy_paste=config['copy_paste'], image_size=(config['max_size'], config['min_size']))
    val_dataset = get_cityscapes_dataset(config['data_root'], 'val')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, config['batch_size']//args.world_size, drop_last=True)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=config['num_workers'], collate_fn=utils.collate_fn
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, sampler=val_sampler, num_workers=config['num_workers'], collate_fn=utils.collate_fn
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True,
            num_workers=config['num_workers'], collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=config['num_workers'], collate_fn=collate_fn)
    print('done', flush=True)

    # Make model
    print('Building model...', end='')
    os.environ['TORCH_HOME'] = TORCH_HOME
    model = maskrcnn_from_config(config)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    print('done')

    # Hook model to wandb
    if args.enable_wandb:
        wandb.watch(model, log="all", log_freq=1000)

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
            state_dict = torch.load(config['checkpoint'], map_location=device)
            model_without_ddp.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
            scaler.load_state_dict(state_dict['scaler'])
            init_epoch = state_dict['epoch'] + 1
            step = state_dict['step']
    elif config['checkpoint'] is not None:
        state_dict = torch.load(config['checkpoint'], map_location=device)
        model_without_ddp.load_state_dict(state_dict['model'])

    # Make metric manager
    metrics = [ScalarMetric(
        'total_loss', log_interval=config['log_interval'], scalar_key='total_loss')]
    for loss_key in LOSS_KEYS:
        metrics.append(ScalarMetric(
            loss_key, log_interval=config['log_interval'], scalar_key=loss_key))
    metrics = MetricManager(
        log_dir, metrics=metrics, purge_step=(step + 1 if step else None), wandb_enabled=args.enable_wandb)

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
            save_checkpoint(ckpt_dir, epoch, step, model_without_ddp, optimizer, lr_scheduler, scaler, args.enable_wandb)
            coco_evaluator = evaluate(model, val_loader, device=device)

            # Log eval metrics
            val_metrics = {}
            for iou_type, coco_eval in coco_evaluator.coco_eval.items():
                mean_ap = coco_eval.stats[0]
                iou_str = 'box_mAP' if iou_type == 'bbox' else 'mask_mAP'
                val_metrics[metrics.val_prefix + iou_str] = mean_ap
                metrics.writer.add_scalar(metrics.val_prefix + iou_str, mean_ap, step)

            metrics.publish_to_wandb(val_metrics, step=step)

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, help='Name of experiment directory')
    parser.add_argument('-r', '--resume', action='store_true', help='Flag to resume training from latest checkpoint')
    parser.add_argument('-ew', '--enable_wandb', action='store_true', help='Enables logging to wandb')
    
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    args = parser.parse_args()

    trainval_cityscapes(args)
