from dataset import get_dataloader
from model import SPHashNet, HashCenter
import transforms
from utils.optim import create_optimizer, cosine_scheduler, exponential_scheduler
from utils.checkpoint import resume, save_checkpoint
from utils.logger import MetricLogger, log_metrics
from utils.loss import calc_loss_l, calc_loss_h
from utils.sacred_ex import ing_base, ing_train, ing_test, parse_config

from sacred import Experiment
from sacred.observers import FileStorageObserver
import argparse
import warnings
import numpy as np
import time

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cfg", "--config", dest="config", type=str)
args = parser.parse_args()

dataset = args.config.split('/')[-1].split('.')[0]
ex_name = 'train_spsvr'
ex = Experiment(ex_name, save_git_info=False, ingredients=[ing_base, ing_train, ing_test])
ex.observers.append(FileStorageObserver(f'log/sacred/{ex_name}/{dataset}'))


@ex.main
def main(_config):
    config = parse_config(_config)
    torch.backends.cudnn.benchmark = True

    # ==================== Dataloader ====================
    train_tfs = nn.Sequential(
        transforms.GroupRandomResizedCrop((224, 224), scale=(0.3, 1.0)),
        transforms.GroupRandomRotation(45),
        transforms.GroupRandomHorizontalFlip(),
        transforms.GroupRandomVerticalFlip(),
        transforms.GroupGaussianBlur(5),
        transforms.GroupConvertImageDtype(torch.float),
        transforms.GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    train_loader = get_dataloader(
        name=config.dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
        data_dir=config.data_dir,
        label_dir=config.label_dir,
        num_class=config.num_class,
        video_names=config.train_names,
        n_frame=config.n_frame,
        duration=config.duration,
        group_tfs=train_tfs
    )
    print(f'training: {len(train_loader.dataset)} samples, {len(train_loader)} batches.')

    # ==================== Model ====================
    net = SPHashNet(code_length=config.code_length, n_frame=config.n_frame)
    model = nn.DataParallel(net).cuda()
    
    centers = HashCenter(n_cls=config.num_class, code_length=config.code_length)
    orth_centers = torch.load(f"data/hash_centers/{config.code_length}_{config.dataset}_{config.num_class}_class.pkl")
    centers.load_state_dict(dict(p=orth_centers))

    # ==================== Optimizer ====================
    # lr
    print(f"LR = {config.lr:.8f}, Min LR = {config.min_lr}")

    # initialize optimizer
    optimizer = create_optimizer(config, model)

    # scheduler
    num_training_steps_per_epoch = len(train_loader)
    lr_schedule_values = exponential_scheduler(
        config.lr, config.scheduler_gamma, config.epochs, num_training_steps_per_epoch, warmup_epochs=config.warmup_epochs
    )
    wd_schedule_values = cosine_scheduler(
        config.weight_decay, config.weight_decay, config.epochs, num_training_steps_per_epoch)

    # ==================== Resume ====================
    if config.checkpoint is not None:
        start_epoch = resume(config.checkpoint, net, optimizer, centers=centers, ex=ex)
        model = nn.DataParallel(net).cuda()
    else:
        start_epoch = 0

    # ==================== Training ====================
    print('Start training.')
    for epoch in range(start_epoch + 1, config.epochs + 1):
        train(model, train_loader, optimizer, lr_schedule_values, wd_schedule_values, centers, epoch, config, ex)
        if epoch % config.save_freq == 0:
            save_checkpoint(f'data/model/{config.dataset}_{epoch}.pth', epoch, model, optimizer, centers=centers)


def train(model, dataloader, optimizer, lr_schedule_values, wd_schedule_values, centers, epoch, config, ex):
    logger = MetricLogger()
    t0 = time.time()
    model.train()
    for i, (_, image_identifiers, x, yp) in enumerate(dataloader):
        # assign learning rate & weight decay for each step
        it = (epoch - 1) * len(dataloader) + i  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for param_group in optimizer.param_groups:
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        image_identifiers = np.array(image_identifiers).T
        x = x.cuda()
        yp = yp.cuda()
        hash_centers = centers.get_centers(device='cuda:0')

        mask_ratio = 0.3 if epoch <= 20 else 0.0
        g, g_h, beh_embedding, x_rec, y_rec, mask = model(x, mask_ratio)
        mask = mask.unsqueeze(2)

        loss_r = mse_loss(x_rec * mask, y_rec * mask)
        loss_h = calc_loss_h(g_h, yp, hash_centers)
        loss_l = calc_loss_l(g, beh_embedding.detach(), yp)
        
        loss = 5 * loss_r if epoch <= 20 else loss_r + 5 * loss_h + loss_l
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses = dict(loss_r=loss_r, loss_h=loss_h, loss_l=loss_l)
        logger.update({"train_loss": losses})

        if (i + 1) % config.print_freq == 0:
            t1 = time.time()
            eta = (len(dataloader) - (i + 1)) / config.print_freq * (t1 - t0)
            log = logger.train_loss
            print(f'[epoch {epoch}] [batch {i + 1}/{len(dataloader)}] '
                  f'[loss_r: {log.loss_r:.4f}, loss_h: {log.loss_h:.4f}, loss_l: {log.loss_l:.4f}] '
                  f'[time {t1-t0:.2f}s, eta {eta:.2f}s]')
            t0 = time.time()

    metrics = logger.get_metrics()
    log_metrics(metrics, epoch, ex)


if __name__ == '__main__':
    ex.run(named_configs=[args.config])
