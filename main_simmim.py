# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np
try:
    import wandb  # Optional; controlled via config.WANDB
except Exception:
    wandb = None

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from timm.utils import AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper

# Native AMP is used; no Apex dependency


def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training (optional; default from env if present)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help='local rank for DistributedDataParallel (defaults to env LOCAL_RANK or 0)'
    )

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    # In pared-down repo, build_loader always returns pre-train loader
    data_loader_train = build_loader(config, logger)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)
    if torch.cuda.is_available():
        device = torch.device('cuda', config.LOCAL_RANK)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    model = model.to(device)
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False
    )
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    # Prefer bf16 on supported GPUs, else fp16
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(torch.cuda.is_available() and amp_dtype == torch.float16))

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, data_loader_train, optimizer, scaler, amp_dtype, epoch, lr_scheduler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, scaler, amp_dtype, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (img, mask, _) in enumerate(data_loader):
        # Measure data loading time (including host->device transfer)
        data_time.update(time.time() - end)
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        with autocast(dtype=amp_dtype):
            loss = model(img, mask)
            loss = loss / max(1, config.TRAIN.ACCUMULATION_STEPS)

        scaler.scale(loss).backward()

        grad_norm = torch.tensor(0.0)
        if (idx + 1) % max(1, config.TRAIN.ACCUMULATION_STEPS) == 0:
            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            optimizer.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step_update(epoch * num_steps + idx)

        # Avoid forcing device/host synchronization here to preserve overlap between
        # data loading and GPU execution. This significantly improves GPU utilization.

        loss_meter.update(loss.item(), img.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            # Log to wandb from rank 0
            if dist.get_rank() == 0 and wandb is not None and wandb.run is not None:
                global_step = epoch * num_steps + idx
                wandb.log({
                    'train/epoch': epoch,
                    'train/iter': idx,
                    'train/global_step': global_step,
                    'train/lr': lr,
                    'train/loss': loss_meter.val,
                    'train/loss_avg': loss_meter.avg,
                    'train/grad_norm': float(norm_meter.val) if hasattr(norm_meter.val, 'item') else norm_meter.val,
                    'train/batch_time': batch_time.val,
                    'train/batch_time_avg': batch_time.avg,
                    'train/data_time': data_time.val,
                    'train/data_time_avg': data_time.avg,
                    'train/memory_mb': memory_used,
                }, step=global_step)
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    _, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = 0
        world_size = 1
    if torch.cuda.is_available():
        torch.cuda.set_device(config.LOCAL_RANK)
        backend = 'nccl'
    else:
        backend = 'gloo'
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    # Allow TF32 where applicable for higher throughput on Ampere+ GPUs
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    # Initialize Weights & Biases on main process if enabled
    if dist.get_rank() == 0 and getattr(config, 'WANDB', None) is not None and config.WANDB.ENABLE:
        if wandb is None:
            logger.warning("WANDB.ENABLE=True but wandb is not installed; proceeding without wandb logging.")
        else:
            wandb_mode = 'offline' if getattr(config.WANDB, 'OFFLINE', False) else None
            run_name = config.WANDB.NAME if getattr(config.WANDB, 'NAME', '') else config.TAG
            try:
                wandb.init(
                    project=config.WANDB.PROJECT,
                    entity=(config.WANDB.ENTITY if getattr(config.WANDB, 'ENTITY', '') else None),
                    name=run_name,
                    dir=config.OUTPUT,
                    mode=wandb_mode,
                    tags=list(getattr(config.WANDB, 'TAGS', [])),
                    notes=getattr(config.WANDB, 'NOTES', ''),
                    config={
                        'model_type': config.MODEL.TYPE,
                        'model_name': config.MODEL.NAME,
                        'img_size': config.DATA.IMG_SIZE,
                        'mask_patch_size': config.DATA.MASK_PATCH_SIZE,
                        'mask_ratio': config.DATA.MASK_RATIO,
                        'batch_size_per_gpu': config.DATA.BATCH_SIZE,
                        'num_workers': config.DATA.NUM_WORKERS,
                        'epochs': config.TRAIN.EPOCHS,
                        'base_lr': config.TRAIN.BASE_LR,
                        'min_lr': config.TRAIN.MIN_LR,
                        'weight_decay': config.TRAIN.WEIGHT_DECAY,
                        'accumulation_steps': config.TRAIN.ACCUMULATION_STEPS,
                    }
                )
                if wandb.run is not None:
                    try:
                        wandb.run.log_code('.')
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")

    try:
        main(config)
    finally:
        if dist.get_rank() == 0 and wandb is not None and getattr(config, 'WANDB', None) is not None and config.WANDB.ENABLE:
            try:
                wandb.finish()
            except Exception:
                pass
