import sys
sys.path.append(".")
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import random
import copy
import open3d as o3d
import math
import torch.nn.functional as F
import pickle
import networkx as nx
random.seed(1234)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class CoronaryPointNoise(Dataset):
    def __init__(self, data_path, point_path, mip_path, split='train', scale_mode='shape_unit', transform=None, point_num=4096, hf_size=[128.0, 168.0, 168.0]):
        super().__init__()
        assert split in ('train', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.data_path = data_path
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform

        self.pointclouds = []
        self.point_path, self.mip_path = point_path, mip_path
        self.grp_path = mip_path[:-1] + '_graph/'

        self.point_num = point_num
        self.hf_size = torch.tensor(hf_size).reshape(1, 3)
        self.get_statistics()
        self.load()

        self.diffusion_steps = 5000
        noise_schedule = "linear"
        betas = get_named_beta_schedule(noise_schedule, self.diffusion_steps)
        betas = np.array(betas, dtype=np.float64)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        assert alphas_cumprod_prev.shape == (num_timesteps,)
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    def get_statistics(self):
        all_points = torch.from_numpy(np.load(self.point_path + f'0000_{self.point_num}.npy')).float()
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0)
        std = all_points.view(-1).std(dim=0)
        self.stats = {'mean': mean, 'std': std}
        return self.stats
        
    def load(self):
        self.name_list = []
        name_list = read_txt(self.data_path + 'split_txt/' + self.split + '.txt')
        for name in name_list:
            self.name_list.append(name + f'_{self.point_num}.pcd')
        print(self.split, len(self.name_list))
        for ii in range(0, len(self.name_list)):
            if ii % 10 == 0:
                print(f'{ii}/{len(self.name_list)}')
            name = self.name_list[ii]

            grp_name = self.grp_path + name[:-9] + '-graph.gpickle'
            fea_name = self.grp_path + name[:-9] + '-label.npy'
            with open(grp_name, 'rb') as f:
                graph = pickle.load(f)
            adj = nx.adjacency_matrix(graph).astype(np.float32).todense()  # (N, N)
            adj = np.array(adj).astype(np.float16)
            fea_lbl = np.load(fea_name)
            # print(fea_lbl.shape)
            geo_fea = fea_lbl[:, :2]

            mip_img = cv2.imread(self.mip_path + name[:-9] + '.png', 0)
            # mip_img = (mip_img - mip_img.min()) / (mip_img.max() - mip_img.min())
            mip_img = mip_img.astype(np.float32) / 127.5 - 1.0
            # img_dist = self.distance_mip(mip_img)
            # print(img_dist.max(), img_dist.min(), (img_dist == 1.0).sum(), (mip_img == 1.0).sum())
            img_dist = torch.from_numpy(mip_img).float()
            img_dist = img_dist[None, :, :]
            img_dist = torch.cat([img_dist, img_dist, img_dist], dim=0)

            point_cloud = o3d.io.read_point_cloud(self.point_path + name)
            pc = torch.from_numpy(np.asarray(point_cloud.points)).float()
            pc_index = (pc * self.hf_size + self.hf_size + 0.5).long()
            if self.scale_mode == 'global_unit':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = self.stats['std'].reshape(1, 1)
            elif self.scale_mode == 'shape_unit':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1)
            elif self.scale_mode == 'shape_half':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1) / (0.5)
            elif self.scale_mode == 'shape_34':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1) / (0.75)
            elif self.scale_mode == 'shape_bbox':
                pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                shift = ((pc_min + pc_max) / 2).view(1, 3)
                scale = (pc_max - pc_min).max().reshape(1, 1) / 2
            else:
                shift = torch.zeros([1, 3])
                scale = torch.ones([1, 1])

            pc = (pc - shift) / scale
            self.pointclouds.append({ 'pointcloud': pc, 'mip_img': img_dist, 'name': name, 'id': ii, 
                                     'shift': shift, 'scale': scale, 'hf_size': self.hf_size, 'pc_index': pc_index,
                                     'adj':torch.from_numpy(adj), 'geo':torch.from_numpy(geo_fea)})
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)
    
    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy.copy(v) for k, v in self.pointclouds[idx].items()}
        mip_img = copy.deepcopy(data['mip_img'])
        # import pdb; pdb.set_trace()
        mip_img = mip_img.unsqueeze(1)
        timestep = torch.randint(0, self.diffusion_steps // 5, (mip_img.shape[0],), dtype=torch.long)
        noise = torch.randn_like(mip_img)
        noisy_img = _extract_into_tensor(self.sqrt_alphas_cumprod, timestep, mip_img.shape) * mip_img \
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timestep, mip_img.shape) * noise
        # noise = torch.randn(mip_img.shape) * 0.08
        # noisy_img = mip_img + noise
        return {
            'pointcloud': data['pointcloud'], 'mip_img': copy.deepcopy(noisy_img.squeeze()), 'name': data['name'], 'id': data['id'], 
            'shift': data['shift'], 'scale': data['scale'], 'hf_size': data['hf_size'], 'pc_index': data['pc_index'], 'adj':data['adj'], 'geo':data['geo']
        }

class CoronaryPointNoiseNew(Dataset):

    def __init__(self, data_path, point_path, mip_path, scale_mode='shape_unit', transform=None, point_num=4096, hf_size=[128.0, 168.0, 168.0]):
        super().__init__()
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.data_path = data_path
        self.scale_mode = scale_mode
        self.transform = transform

        self.pointclouds = []
        self.point_path, self.mip_path = point_path, mip_path
        self.grp_path = mip_path[:-1] + '_graph/'

        self.point_num = point_num
        self.hf_size = torch.tensor(hf_size).reshape(1, 3)
        self.get_statistics()
        self.load()

        self.diffusion_steps = 5000
        noise_schedule = "linear"
        betas = get_named_beta_schedule(noise_schedule, self.diffusion_steps)
        betas = np.array(betas, dtype=np.float64)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        assert alphas_cumprod_prev.shape == (num_timesteps,)
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    def get_statistics(self):
        all_points = torch.from_numpy(np.load(self.point_path + f'0000_{self.point_num}.npy')).float()
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0)
        std = all_points.view(-1).std(dim=0)
        self.stats = {'mean': mean, 'std': std}
        return self.stats
        
    def load(self):
        self.name_list = []
        name_list = sorted(os.listdir(self.mip_path))
        for name in name_list:
            if '.png' in name:
                self.name_list.append(name[:-4])
        for ii in range(0, len(self.name_list)):
            if ii % 10 == 0:
                print(f'{ii}/{len(self.name_list)}')
            name = self.name_list[ii]

            grp_name = self.grp_path + name + '-graph.gpickle'
            fea_name = self.grp_path + name + '-label.npy'
            with open(grp_name, 'rb') as f:
                graph = pickle.load(f)
            adj = nx.adjacency_matrix(graph).astype(np.float32).todense()  # (N, N)
            adj = np.array(adj).astype(np.float16)
            fea_lbl = np.load(fea_name)
            # print(fea_lbl.shape)
            geo_fea = fea_lbl[:, :2]

            mip_img = cv2.imread(self.mip_path + name + '.png', 0)
            mip_img = mip_img.astype(np.float32) / 127.5 - 1.0
            img_dist = torch.from_numpy(mip_img).float()
            img_dist = img_dist[None, :, :]
            img_dist = torch.cat([img_dist, img_dist, img_dist], dim=0)

            point_cloud = o3d.io.read_point_cloud(self.point_path + 'Normal_1_4096.pcd')
            pc = torch.from_numpy(np.asarray(point_cloud.points)).float()
            pc_index = (pc * self.hf_size + self.hf_size + 0.5).long()
            if self.scale_mode == 'global_unit':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = self.stats['std'].reshape(1, 1)
            elif self.scale_mode == 'shape_unit':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1)
            elif self.scale_mode == 'shape_half':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1) / (0.5)
            elif self.scale_mode == 'shape_34':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1) / (0.75)
            elif self.scale_mode == 'shape_bbox':
                pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                shift = ((pc_min + pc_max) / 2).view(1, 3)
                scale = (pc_max - pc_min).max().reshape(1, 1) / 2
            else:
                shift = torch.zeros([1, 3])
                scale = torch.ones([1, 1])

            pc = (pc - shift) / scale
            self.pointclouds.append({ 'pointcloud': pc, 'mip_img': img_dist, 'name': name, 'id': ii, 
                                     'shift': shift, 'scale': scale, 'hf_size': self.hf_size, 'pc_index': pc_index,
                                     'adj':torch.from_numpy(adj), 'geo':torch.from_numpy(geo_fea)})
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)
    
    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy.copy(v) for k, v in self.pointclouds[idx].items()}
        mip_img = copy.deepcopy(data['mip_img'])
        # import pdb; pdb.set_trace()
        mip_img = mip_img.unsqueeze(1)
        timestep = torch.randint(0, self.diffusion_steps // 10, (mip_img.shape[0],), dtype=torch.long)
        noise = torch.randn_like(mip_img)
        noisy_img = _extract_into_tensor(self.sqrt_alphas_cumprod, timestep, mip_img.shape) * mip_img \
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timestep, mip_img.shape) * noise
        noisy_img = F.interpolate(noisy_img, size=(336, 336), mode='bilinear', align_corners=False)
        # noise = torch.randn(mip_img.shape) * 0.08
        # noisy_img = mip_img + noise
        return {
            'pointcloud': data['pointcloud'], 'mip_img': copy.deepcopy(noisy_img.squeeze()), 'name': data['name'], 'id': data['id'], 
            'shift': data['shift'], 'scale': data['scale'], 'hf_size': data['hf_size'], 'pc_index': data['pc_index'], 'adj':data['adj'], 'geo':data['geo']
        }

def read_txt(file_name=None):
    if file_name is None:
        return None
    name_list = []
    f = open(file_name, 'r')
    a = f.readlines()
    for name in a:
        name_list.append(name[:-1])
    return name_list
