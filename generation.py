import os
from contextlib import nullcontext
from pathlib import Path
import hydra
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from utils_fun import training_utils
from utils_fun.dataset import CoronaryPointNoise, CoronaryPointNoiseNew
from model import get_model, Two2ThreeDiffusionModel
from config.structured import ProjectConfig
import matplotlib.pyplot as plt
import open3d as o3d
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path='config', config_name='config', version_base='1.1')
def main(cfg: ProjectConfig):
    print(cfg.run)
    # Accelerator
    accelerator = Accelerator(mixed_precision=cfg.run.mixed_precision, cpu=cfg.run.cpu, gradient_accumulation_steps=cfg.optimizer.gradient_accumulation_steps)
    training_utils.setup_distributed_print(accelerator.is_main_process)
    
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

    # Dataset
    print(cfg.checkpoint.resume)
    train_state: training_utils.TrainState = training_utils.resume_from_checkpoint(cfg, model, optimizer, scheduler)
    if cfg.dataset.test_origin == 'real':
        test_set = CoronaryPointNoise(cfg.dataset.root, cfg.dataset.root + 'coronary3d_point/', cfg.dataset.root + 'same_mip_all/', split='test')
    elif cfg.dataset.test_origin == 'generated':
        test_set = CoronaryPointNoiseNew(cfg.dataset.root, cfg.dataset.root + 'coronary3d_point/', cfg.dataset.root + 'generated2d/mip_img/')
    else:
        return
    dataloader_test = DataLoader(dataset=test_set, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.num_workers, shuffle=False)
    model, optimizer, scheduler, dataloader_test = accelerator.prepare(model, optimizer, scheduler, dataloader_test)

    model: Two2ThreeDiffusionModel
    optimizer: torch.optim.Optimizer
    sample_context = nullcontext
    with sample_context():
        model_infer(
            cfg=cfg,
            model=model,
            dataloader=dataloader_test,
            accelerator=accelerator,
        )
    

@torch.no_grad()
def model_infer(cfg, model, dataloader, accelerator, output_dir='sample',):
    model.eval()
    output_dir: Path = Path(output_dir)
    for batch_idx, batch in enumerate(dataloader):
        # Sample
        pc, mip_img, mean = batch['pointcloud'].float(), batch['mip_img'].float(), batch['shift'].float()
        std, hf_size = batch['scale'].float(), batch['hf_size'].float()
        adj, geo = batch['adj'].float(), batch['geo'].float()
            
        output, all_outputs = model(pc, mip_img, mean, std, hf_size, mode='sample', adj=adj, geo=geo, return_sample_every_n_steps=4, 
            num_inference_steps=cfg.run.num_inference_steps, disable_tqdm=(not accelerator.is_main_process))
        for ii in range(0, output.shape[0]):
            point_pred, point_lbl = output[ii].detach().cpu().numpy(), pc[ii].detach().cpu().numpy()
            sample_all = all_outputs[ii].detach().cpu().numpy()
            x1, y1 = point_lbl[:, 1], point_lbl[:, 2]
            x2, y2 = point_pred[:, 1] + (point_lbl[:, 1].max() - point_pred[:, 1].min() + 0.1), point_pred[:, 2]
            plt.scatter(x1, y1, s=10, c='r')
            plt.scatter(x2, y2, s=10, c='g')
            plt.axis("scaled")
            plt.xlabel("X Axis")
            plt.ylabel("Y Axis")
            plt.title("2D Point Cloud Visualization")
            plt.savefig(batch['name'][ii] + '.png')
            plt.close()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_pred)
            o3d.io.write_point_cloud(batch['name'][ii] + '.ply', pcd)
            # np.savez(batch['name'][ii] + '.npz', sample_all)
            

if __name__ == '__main__':
    main()