import datetime
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable
import hydra
import torch
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from utils_fun import training_utils
from utils_fun.dataset import CoronaryPointNoise
from model import get_model, ConditionalPointCloudDiffusionModel
from config.structured import ProjectConfig
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')

@hydra.main(config_path='config', config_name='config', version_base='1.1')
def main(cfg: ProjectConfig):
    print(cfg.run)
    # Accelerator
    accelerator = Accelerator(mixed_precision=cfg.run.mixed_precision, cpu=cfg.run.cpu, gradient_accumulation_steps=cfg.optimizer.gradient_accumulation_steps)

    # Logging
    training_utils.setup_distributed_print(accelerator.is_main_process)
    if cfg.logging.wandb and accelerator.is_main_process:
        wandb.init(project=cfg.logging.wandb_project, name=cfg.run.name, job_type=cfg.run.job, 
                   config=OmegaConf.to_container(cfg))
        wandb.run.log_code(root=hydra.utils.get_original_cwd(),
            include_fn=lambda p: any(p.endswith(ext) for ext in ('.py', '.json', '.yaml', '.md', '.txt.', '.gin')),
            exclude_fn=lambda p: any(s in p for s in ('output', 'tmp', 'wandb', '.git', '.vscode')))
        cfg: ProjectConfig = DictConfig(wandb.config.as_dict())  # get the config back from wandb for hyperparameter sweeps

    # Configuration
    print(f'Current working directory: {os.getcwd()}')
    log_info_txt = open('log.txt', 'w')
    log_info_txt.write(OmegaConf.to_yaml(cfg))
    log_info_txt.flush()
    print(OmegaConf.to_yaml(cfg))
    # import pdb; pdb.set_trace()
    # Set random seed
    training_utils.set_seed(cfg.run.seed)

    # Model
    model = get_model(cfg)
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')

    # Exponential moving average of model parameters
    if cfg.ema.use_ema:
        from torch_ema import ExponentialMovingAverage
        model_ema = ExponentialMovingAverage(model.parameters(), decay=cfg.ema.decay)
        model_ema.to(accelerator.device)
        print('Initialized model EMA')
    else:
        model_ema = None
        print('Not using model EMA')

    # Optimizer and scheduler
    optimizer = training_utils.get_optimizer(cfg, model, accelerator)
    scheduler = training_utils.get_scheduler(cfg, optimizer)

    # Resume from checkpoint and create the initial training state
    train_state: training_utils.TrainState = training_utils.resume_from_checkpoint(cfg, model, optimizer, scheduler, model_ema)

    # Datasets
    test_set = CoronaryPointNoise(cfg.dataset.root, cfg.dataset.root + 'coronary3d_point/', cfg.dataset.root + 'same_mip_all/', split='test')
    train_set = CoronaryPointNoise(cfg.dataset.root, cfg.dataset.root + 'coronary3d_point/', cfg.dataset.root + 'same_mip_all/', split='train')
    dataloader_test = DataLoader(dataset=test_set, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.num_workers, shuffle=False)
    dataloader_train = DataLoader(dataset=train_set, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.num_workers, shuffle=True)
    # Compute total training batch size
    total_batch_size = cfg.dataloader.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    model, optimizer, scheduler, dataloader_train, dataloader_test = accelerator.prepare(
        model, optimizer, scheduler, dataloader_train, dataloader_test)
    model: ConditionalPointCloudDiffusionModel
    optimizer: torch.optim.Optimizer

    ss = f'\n***** Starting training at {datetime.datetime.now()} *****\n'
    ss += f'    Dataset train size: {len(dataloader_train.dataset):_}\n'
    ss += f'    Dataset val size: {len(dataloader_train.dataset):_}\n'
    ss += f'    Dataloader train size: {len(dataloader_train):_}\n'
    ss += f'    Batch size per device = {cfg.dataloader.batch_size}\n'
    ss += f'    Total train batch size (w. parallel, dist & accum) = {total_batch_size}\n'
    ss += f'    Gradient Accumulation steps = {cfg.optimizer.gradient_accumulation_steps}\n'
    ss += f'    Max training steps = {cfg.run.max_steps}\n'
    ss += f'    Training state = {train_state}\n'
    log_info_txt.write(ss)
    log_info_txt.flush()
    
    # Info
    print(f'***** Starting training at {datetime.datetime.now()} *****')
    print(f'    Dataset train size: {len(dataloader_train.dataset):_}')
    print(f'    Dataset val size: {len(dataloader_train.dataset):_}')
    print(f'    Dataloader train size: {len(dataloader_train):_}')
    print(f'    Batch size per device = {cfg.dataloader.batch_size}')
    print(f'    Total train batch size (w. parallel, dist & accum) = {total_batch_size}')
    print(f'    Gradient Accumulation steps = {cfg.optimizer.gradient_accumulation_steps}')
    print(f'    Max training steps = {cfg.run.max_steps}')
    print(f'    Training state = {train_state}')

    while True:
    
        log_header = f'Epoch: [{train_state.epoch}]'
        metric_logger = training_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('step', training_utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
        metric_logger.add_meter('lr', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        progress_bar: Iterable[Any] = metric_logger.log_every(dataloader_train, cfg.run.print_step_freq, 
            header=log_header)

        # Train
        for i, batch in enumerate(progress_bar):
            if (cfg.run.limit_train_batches is not None) and (i >= cfg.run.limit_train_batches): break
            model.train()
            pc, mip_img, mean = batch['pointcloud'].float(), batch['mip_img'].float(), batch['shift'].float()
            std, hf_size = batch['scale'].float(), batch['hf_size'].float()
            adj, geo = batch['adj'].float(), batch['geo'].float()
            # Gradient accumulation
            with accelerator.accumulate(model):

                # Forward
                loss = model(pc, mip_img, mean, std, hf_size, mode='train', adj=adj, geo=geo)

                # Backward
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # grad_norm_unclipped = training_utils.compute_grad_norm(model.parameters())  # useless w/ mixed prec
                    if cfg.optimizer.clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.clip_grad_norm)
                    grad_norm_clipped = training_utils.compute_grad_norm(model.parameters())

                # Step optimizer
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    scheduler.step()
                    train_state.step += 1

                # Exit if loss was NaN
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

            # Gradient accumulation
            if accelerator.sync_gradients:

                # Logging
                log_dict = {
                    'lr': optimizer.param_groups[0]["lr"],
                    'step': train_state.step,
                    'train_loss': loss_value,
                    # 'grad_norm_unclipped': grad_norm_unclipped,  # useless w/ mixed prec
                    'grad_norm_clipped': grad_norm_clipped,
                }
                metric_logger.update(**log_dict)
                if accelerator.is_main_process and train_state.step % cfg.run.log_step_freq == 0:
                    step_s = str(train_state.step).zfill(7)
                    ss = '\nstep: ' + step_s + ',  '
                    ss = f'lr: {optimizer.param_groups[0]["lr"]}, train_loss:{loss_value} \n'
                    log_info_txt.write(ss)
                    log_info_txt.flush()
                if (cfg.logging.wandb and accelerator.is_main_process and train_state.step % cfg.run.log_step_freq == 0):
                    wandb.log(log_dict, step=train_state.step)
            
                # Update EMA
                if cfg.ema.use_ema and train_state.step % cfg.ema.update_every == 0:
                    model_ema.update(model.parameters())

                # Save a checkpoint
                if accelerator.is_main_process and (train_state.step % cfg.run.checkpoint_freq == 0 or train_state.step == 1):
                    
                    checkpoint_dict = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': train_state.epoch,
                        'step': train_state.step,
                        'best_val': train_state.best_val,
                        'model_ema': model_ema.state_dict() if model_ema else {},
                        'cfg': cfg
                    }
                    checkpoint_path = f'checkpoint-{train_state.step}.pth'
                    accelerator.save(checkpoint_dict, checkpoint_path)
                    print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')

                # End training after the desired number of steps/epochs
                if train_state.step >= cfg.run.max_steps:
                    print(f'Ending training at: {datetime.datetime.now()}')
                    print(f'Final train state: {train_state}')
                    ss = f'Ending training at: {datetime.datetime.now()}\n'
                    ss += f'Final train state: {train_state}'
                    log_info_txt.write(ss)
                    log_info_txt.flush()
                    log_info_txt.close()
                    
                    # wandb.finish()
                    # time.sleep(5)
                    return

        # Epoch complete, log it and continue training
        train_state.epoch += 1

        # Gather stats from all processes
        metric_logger.synchronize_between_processes(device=accelerator.device)
        print(f'{log_header}  Average stats --', metric_logger)
   
if __name__ == '__main__':
    main()