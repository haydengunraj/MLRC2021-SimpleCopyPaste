import os
import torch
from detection.utils import collate_fn
from detection.engine import train_one_epoch, evaluate

from model import mask_rcnn
from data import get_cityscapes_dataset, NUM_CLASSES

TORCH_HOME = 'weights'


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


def trainval_cityscapes(cityscapes_root, output_dir, epochs=10, batch_size=2, lr=0.005, momentum=0.9, weight_decay=0.0005):
    # Set device and clear GPU cache
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    # Make datasets and data loaders
    print('Building datasets...', end='', flush=True)
    train_dataset, val_dataset = get_cityscapes_dataset(cityscapes_root)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4,
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
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    print('done')

    # Main training/val loop
    print('Starting training')
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10, scaler=scaler)
        lr_scheduler.step()
        evaluate(model, val_loader, device=device)
        save_checkpoint(output_dir, epoch, model, optimizer, lr_scheduler, scaler)

    return model


if __name__ == '__main__':
    root = 'D:\\Datasets\\Cityscapes'
    out_dir = 'test_ckpts'
    epochs = 50
    batch_size = 1
    lr = 0.005
    momentum = 0.9
    weight_decay = 0.0005

    trained_model = trainval_cityscapes(
        root, out_dir, epochs, batch_size, lr, momentum, weight_decay)
