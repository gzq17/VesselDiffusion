import torch
import numpy as np
import networkx as nx
import skfmm
import pickle
import os
import cv2
from scipy.ndimage import zoom
import copy
import math
import argparse

def generate_edge(graph, img, edge_dist_thresh, pre_th):
    img_x, img_y = img.shape[0], img.shape[1]
    speed = img
    node_list = list(graph.nodes)
    num_node = len(node_list)
    num_edge = 0
    for i, n in enumerate(node_list):
        if speed[graph.nodes[n]['x'], graph.nodes[n]['y']] < 0.1:
            continue
        neighbor = speed[max(0, graph.nodes[n]['x'] - 1):min(img_x, graph.nodes[n]['x'] + 2),
                    max(0, graph.nodes[n]['y'] - 1):min(img_y, graph.nodes[n]['y'] + 2)]
        if np.mean(neighbor) < pre_th:
            continue
        phi = np.ones_like(speed)
        phi[graph.nodes[n]['x'], graph.nodes[n]['y']] = -1.0
        try:
                # this commonly fails in skfmm due to numerics issues
            tt = skfmm.travel_time(phi=phi, speed=speed, order=2)
        except:
            # fall back to order=1
            tt = skfmm.travel_time(phi=phi, speed=speed, order=1)
        for n_comp in node_list[i+1:]:
            geo_dist = tt[graph.nodes[n_comp]['x'], graph.nodes[n_comp]['y']]  # travel time
            if geo_dist < edge_dist_thresh:
                graph.add_edge(n, n_comp, weight=edge_dist_thresh/(edge_dist_thresh + geo_dist))
                num_edge += 1
    print('Generate total', num_node, 'nodes, ', num_edge, 'edges.')
    return graph

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
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
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def generate_by_noise_lbl(lbl_path='./data/same_mip_all/', true_lbl=True):
    out_path = lbl_path[:-1] + '_graph/'
    os.makedirs(out_path, exist_ok=True)
    name_list = sorted(os.listdir(lbl_path), reverse=False)
    sz = 6
    edge_dist_thresh = 8 * sz / 1.5
    xx = np.arange(0, 336, sz) + sz // 2
    yy = np.arange(0, 336, sz) + sz // 2
    X, Y = np.meshgrid(xx, yy)
    point_arr = np.vstack([Y.ravel(), X.ravel()]).T
    kk = 1
    
    diffusion_steps = 5000
    noise_schedule = "linear"
    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    betas = np.array(betas, dtype=np.float64)
    num_timesteps = int(betas.shape[0])
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    assert alphas_cumprod_prev.shape == (num_timesteps,)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    
    for name in name_list:
        if '.png' not in name:
            continue
        print('now:{}/{}'.format(kk, len(name_list)), name)
        graph_save_path = out_path + name[:-4] + "-graph.gpickle"
        if os.path.exists(graph_save_path):
            continue
        lbl_name = lbl_path + name
        lbl = cv2.imread(lbl_name, 0)
        if true_lbl:
            lbl[lbl == 255] = 1
        else:
            lbl = (lbl - lbl.min()) / (lbl.max() - lbl.min())
            sz1 = lbl.shape[0]
            lbl = zoom(lbl, (336/sz1, 336/sz1), order=1)
            print(lbl.shape, lbl.min(), lbl.max())
        lbl = lbl.astype('float')
        point_value = lbl[point_arr[:, 0], point_arr[:, 1]]
        pos_fea = copy.deepcopy(point_arr)
        pos_fea = (pos_fea - 168) / 168
        fea_label = np.concatenate([pos_fea, point_value[:, np.newaxis]], axis=1)
        print(point_value.shape, fea_label.shape)
        np.save(out_path + name[:-4] + '-label.npy', fea_label)
        
        if true_lbl:
            img_dist = copy.deepcopy(lbl)
            img_dist[img_dist == 0] = -1.0
            img_dist_th = torch.from_numpy(img_dist).unsqueeze(0).unsqueeze(0).float()
            timestep = torch.tensor([100], dtype=torch.long, device='cpu')
            noise = torch.randn_like(img_dist_th)
            noisy_img = _extract_into_tensor(sqrt_alphas_cumprod, timestep, img_dist_th.shape) * img_dist_th \
                + _extract_into_tensor(sqrt_one_minus_alphas_cumprod, timestep, img_dist_th.shape) * noise
            print(noisy_img.shape, noisy_img.max(), noisy_img.min())
            noisy_img = noisy_img.squeeze().numpy()
            img_dist = (noisy_img - noisy_img.min()) / (noisy_img.max() - noisy_img.min())
        else:
            img_dist = copy.deepcopy(lbl)
        print(img_dist.shape, img_dist.max(), img_dist.min())
        graph = nx.Graph()
        for ii in range(0, point_arr.shape[0]):
            graph.add_node(ii, kind='MP', x=point_arr[ii, 0], y=point_arr[ii, 1], label=ii)
        # graph = generate_edge(graph, img_dist, edge_dist_thresh, 0.15)
        graph = generate_edge(graph, img_dist, edge_dist_thresh, 0.1)#0.1
        with open(graph_save_path, 'wb') as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
        graph.clear()
        kk += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data/same_mip_all/')
    parser.add_argument('--lbl', type=int, default=1)
    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = parse_args()
    print(opt)
    true_lbl = True
    if opt.lbl != 1:
        true_lbl=False
    generate_by_noise_lbl(opt.path, true_lbl)
