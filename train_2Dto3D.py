import datetime
import os
from pathlib import Path
from typing import Any, Iterable
import hydra
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from utils_fun import training_utils
from utils_fun.dataset import CoronaryPointNoise
from model import get_model, Two2ThreeDiffusionModel
from config.structured import ProjectConfig
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')

@hydra.main(config_path='config', config_name='config', version_base='1.1')
def main(cfg: ProjectConfig):
    accelerator = Accelerator(mixed_precision=cfg.run.mixed_precision, cpu=cfg.run.cpu, gradient_accumulation_steps=cfg.optimizer.gradient_accumulation_steps)
    print(f'Current working directory: {os.getcwd()}')
    log_info_txt = open('log.txt', 'w')
    log_info_txt.write(OmegaConf.to_yaml(cfg))
    log_info_txt.flush()
    print(OmegaConf.to_yaml(cfg))
    training_utils.set_seed(cfg.run.seed)

    # Model
    model = get_model(cfg)
    optimizer = training_utils.get_optimizer(cfg, model, accelerator)
    scheduler = training_utils.get_scheduler(cfg, optimizer)
    train_state: training_utils.TrainState = training_utils.resume_from_checkpoint(cfg, model, optimizer, scheduler)

    # Datasets
    test_set = CoronaryPointNoise(cfg.dataset.root, cfg.dataset.root + 'coronary3d_point/', cfg.dataset.root + 'same_mip_all/', split='test')
    train_set = CoronaryPointNoise(cfg.dataset.root, cfg.dataset.root + 'coronary3d_point/', cfg.dataset.root + 'same_mip_all/', split='train')
    dataloader_test = DataLoader(dataset=test_set, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.num_workers, shuffle=False)
    dataloader_train = DataLoader(dataset=train_set, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.num_workers, shuffle=True)
    total_batch_size = cfg.dataloader.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    model, optimizer, scheduler, dataloader_train, dataloader_test = accelerator.prepare(model, optimizer, scheduler, dataloader_train, dataloader_test)
    model: Two2ThreeDiffusionModel
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
    print(ss)

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
            with accelerator.accumulate(model):
                loss = model(pc, mip_img, mean, std, hf_size, mode='train', adj=adj, geo=geo)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if cfg.optimizer.clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.clip_grad_norm)
                    grad_norm_clipped = training_utils.compute_grad_norm(model.parameters())

                # Step optimizer
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    scheduler.step()
                    train_state.step += 1
                loss_value = loss.item()

            # Gradient accumulation
            if accelerator.sync_gradients:
                log_dict = {
                    'lr': optimizer.param_groups[0]["lr"],
                    'step': train_state.step,
                    'train_loss': loss_value,
                    'grad_norm_clipped': grad_norm_clipped,
                }
                metric_logger.update(**log_dict)
                if accelerator.is_main_process and train_state.step % cfg.run.log_step_freq == 0:
                    step_s = str(train_state.step).zfill(7)
                    ss = '\nstep: ' + step_s + ',  '
                    ss = f'lr: {optimizer.param_groups[0]["lr"]}, train_loss:{loss_value} \n'
                    log_info_txt.write(ss)
                    log_info_txt.flush()
                # Save model
                if accelerator.is_main_process and (train_state.step % cfg.run.checkpoint_freq == 0 or train_state.step == 1):     
                    checkpoint_dict = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': train_state.epoch,
                        'step': train_state.step,
                        'best_val': train_state.best_val,
                        'cfg': cfg
                    }
                    checkpoint_path = f'checkpoint-{train_state.step}.pth'
                    accelerator.save(checkpoint_dict, checkpoint_path)
                    print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')

                # End training after the desired number of steps/epochs
                if train_state.step >= cfg.run.max_steps:
                    log_info_txt.flush()
                    log_info_txt.close()
                    return
        train_state.epoch += 1
        metric_logger.synchronize_between_processes(device=accelerator.device)
        print(f'{log_header}  Average stats --', metric_logger)
   
if __name__ == '__main__':
    main()