from model.feature_model import FeatureModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import copy
import pickle
import networkx as nx
from accelerate import Accelerator
import math
import argparse

LR = 5e-4
MAX_EPOCH = 1000
BS = 8

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

class PreTrainDataNoise(Dataset):
    
    def __init__(self, parent_path, img_path='img/', lbl_path='lbl/', grp_path='lbl_graph/',
                 split='train'):
        self.test_list = ['0_001_0.png', '1_000350000000_mra_0.png',]
        self.img_path = parent_path + img_path
        self.lbl_path = parent_path + lbl_path
        self.grp_path = parent_path + grp_path
        self.split = split
        self.img_name_list = sorted(os.listdir(self.img_path))
        self.img_list = self.img_load(self.img_name_list)

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
    
    def img_load(self, name_list):
        img_list = []
        num_list = np.zeros((3,))
        kk = 0
        for name in name_list:
            if self.split == 'train' and name in self.test_list:
                continue
            if self.split == 'test' and name not in self.test_list:
                continue
            if kk % 20 == 0:
                print(kk, name)
            kk += 1
            label = int(name[0])
            num_list[label] += 1
            img_name = self.img_path + name
            lbl_name = self.lbl_path + name.replace('.png', '_label.png')
            grp_name = self.grp_path + name.replace('.png', '_label-graph.gpickle')
            fea_name = self.grp_path + name.replace('.png', '_label-label.npy')
            with open(grp_name, 'rb') as f:
                graph = pickle.load(f)
            adj = nx.adjacency_matrix(graph).astype(np.float32).todense()  # (N, N)
            adj = np.array(adj).astype(np.float16)
            fea_lbl = np.load(fea_name)
            # print(fea_lbl.shape)
            geo_fea = fea_lbl[:, :2]
            grp_lbl = fea_lbl[:, 2]
            img = cv2.imread(img_name, 0)
            img = (img - img.min()) / (img.max() - img.min()) * 2 - 1.0
            img = torch.from_numpy(img).float()
            img = img[None, :, :]
            img = torch.cat([img, img, img], dim=0)

            lbl = cv2.imread(lbl_name, 0)
            lbl[lbl == 255] = 1
            lbl = torch.from_numpy(lbl).float()
            
            img_list.append({'img':img, 'lbl':lbl, 'cls':torch.tensor([label]), 
                             'adj':torch.from_numpy(adj), 'geo':torch.from_numpy(geo_fea),
                             'grp_lbl':torch.from_numpy(grp_lbl)})
            # print(name, img.max(), img.min(), lbl.max(), lbl.min())
        print(num_list)
        return img_list
    
    def __len__(self,):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy.copy(v) for k, v in self.img_list[idx].items()}
        mip_img = copy.deepcopy(data['img'])
        mip_img = mip_img.unsqueeze(1)
        timestep = torch.randint(0, self.diffusion_steps // 10, (mip_img.shape[0],), dtype=torch.long)
        noise = torch.randn_like(mip_img)
        noisy_img = _extract_into_tensor(self.sqrt_alphas_cumprod, timestep, mip_img.shape) * mip_img \
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timestep, mip_img.shape) * noise
        noisy_img = F.interpolate(noisy_img, size=(336, 336), mode='bilinear', align_corners=False)

        return {
            'img':noisy_img[:, 0, :, :], 'lbl':data['lbl'], 'cls':data['cls'], 
                             'adj':data['adj'], 'geo':data['geo'],
                             'grp_lbl':data['grp_lbl']
        }

class GraphAttentionLayer(nn.Module):
    
    def __init__(self, in_features, out_features, dropout=0, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.fc_W = nn.Linear(in_features, out_features)
        self.fc_a = nn.Linear(2 * out_features, 1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # import pdb; pdb.set_trace()
        hh = self.fc_W(input) 
        B = hh.size()[0] 
        N = hh.size()[1]

        a_input = torch.cat([hh.repeat(1, 1, N).view(B, N * N, -1), hh.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_features)
        e = self.leakyrelu(self.fc_a(a_input).squeeze(3))

        zero_vec = -9e3*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, hh)

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT3D(nn.Module):
    def __init__(self, n_feature, n_hidden, dropout=0, alpha=0.2, n_heads=1): # n_classes
        super(GAT3D, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(n_feature, n_hidden, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(n_hidden * n_heads, n_classes, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):

        x_feat = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        return x_feat
        # x = self.out_att(x_feat, adj)
        # return torch.sigmoid(x), x_feat  # (B, N, n_classes), (B, N, n_hidden * n_heads)

class DualStreamFE(nn.Module):

    def __init__(self, GAT_nhidden=8, GAT_nhead=4, dropout=0, img_sz=336, n_class=6):
        super(DualStreamFE, self).__init__()

        ###deep features:
        self.feature_model = FeatureModel(image_size=img_sz, model_name='vit_small_patch16_224_msn')

        self.stage1_seg_fea = nn.Sequential(nn.Conv2d(384, 32, kernel_size=3, stride=1, padding=1),
                                        nn.InstanceNorm2d(32),)
        self.stage1_seg = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1, stride=1),
                                        nn.Sigmoid())
        
        self.cls_head = nn.Sequential(nn.Linear(384, 32), nn.ReLU(),
                                      nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())
        self.stage1_cls = nn.Sequential(nn.Linear(32, n_class),
                                        nn.Softmax(dim=-1))
        

        self.conv_fea = nn.Sequential(nn.Conv2d(384, 32, kernel_size=1, stride=1),
                                        nn.InstanceNorm2d(40))
        ### geometric features:
        self.fc_geofeature = nn.Sequential(nn.Linear(2, 8), nn.ReLU(),
                                         nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())
        
        ### CGAT:
        self.CGAT_Layer1 = GAT3D(n_feature = 40, n_hidden=GAT_nhidden, n_heads=GAT_nhead, dropout=dropout)
        self.CGAT_Layer2 = GAT3D(n_feature = GAT_nhidden*GAT_nhead, n_hidden=GAT_nhidden, n_heads=GAT_nhead, dropout=dropout)
        # self.CGAT_Layer3 = GAT3D(n_feature = GAT_nhidden*GAT_nhead, n_hidden=GAT_nhidden, n_heads=GAT_nhead, dropout=dropout)
        self.stage2_seg_fea = nn.Sequential(nn.Linear((GAT_nhidden * GAT_nhead), 32), nn.ReLU(),
                                          nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())    
        
        self.stage2_cls_fea = nn.Sequential(nn.Linear((GAT_nhidden * GAT_nhead), 32), nn.ReLU(),
                                          nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())
        self.stage2_cls = nn.Sequential(nn.Linear(32, n_class),
                                        nn.Softmax(dim=-1))
        ### output:
        self.output_layer = nn.Sequential(nn.Linear(32, 32), nn.ReLU(),
                                          nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                                          nn.Linear(32, 2),
                                          nn.Softmax(dim=-1))
        
        self.feature_dim = 32

    def forward(self, img, adj=None, geometric_features=None, return_type='result'):
        output_cls, output_feats = self.feature_model(img, return_type='all')
        # import pdb;pdb.set_trace()
        stage1_seg_fea = self.stage1_seg_fea(output_feats)
        stage1_seg = self.stage1_seg(stage1_seg_fea)
        stage1_cls_fea = self.cls_head(output_cls)
        stage1_cls = self.stage1_cls(stage1_cls_fea)
        
        deep_feature_one = self.conv_fea(output_feats)
        deep_feature = F.interpolate(deep_feature_one, size=(56, 56), mode='bilinear', align_corners=False)
        N, C, _, _ = deep_feature.size()
        deep_fea = deep_feature.view(N, C, -1).transpose(1, 2).contiguous()
        geo_fea = self.fc_geofeature(geometric_features)
        vertex_features = torch.cat([deep_fea, geo_fea], dim=-1)
        
        x = vertex_features
        # import pdb; pdb.set_trace()
        x_feat_1 = self.CGAT_Layer1(x, adj)
        x_feat_2 = self.CGAT_Layer2(x_feat_1, adj)
        # x_feat_3 = self.CGAT_Layer3(x_feat_2, adj) # [1,n,64]
        stage2_seg_fea = self.stage2_seg_fea(x_feat_2)
        stage2_seg = self.output_layer(stage2_seg_fea)

        stage2_cls_fea = self.stage2_cls_fea(torch.mean(x_feat_2, dim=1))
        stage2_cls = self.stage2_cls(stage2_cls_fea)

        if return_type == 'result':
            return stage1_seg, stage1_cls, stage2_seg, stage2_cls
        elif return_type == 'feature':
            # import pdb; pdb.set_trace()
            C = stage2_seg_fea.shape[2]
            gcn_fea = stage2_seg_fea.transpose(1, 2).contiguous().view(N, C, 56, 56)
            gcn_fea = F.interpolate(gcn_fea, size=(336, 336), mode='bilinear', align_corners=False)
            return stage1_cls_fea, stage1_seg_fea, stage2_cls_fea, gcn_fea

def dice_loss(y_pred, y_true):
    ndims = len(list(y_pred.size())) - 2
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom)
    return 1.0 - dice

def adjust_lr(optimizer, epoch, lr):
    lr_c = lr * ((1 - epoch/(MAX_EPOCH + 1)) ** 0.9)
    for p in optimizer.param_groups:
        p['lr'] = lr_c

def main(data_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16', cpu=False, gradient_accumulation_steps=1)
    loss_f = open(out_path + 'loss.txt', 'w')
    model = DualStreamFE().cuda()
    model.train()
    trainset = PreTrainDataNoise(data_path, split='train')
    train_loader = DataLoader(trainset, batch_size=BS, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.00001)
    ce_weight = torch.zeros((6, )) + 1.0
    ce_weight[-1] = 0.1
    criterion_cls = nn.CrossEntropyLoss(weight=ce_weight.cuda())
    
    seg_weight = torch.zeros((2, )) + 1.0
    ce_weight[0] = 0.1
    criterion_seg = nn.CrossEntropyLoss(weight=seg_weight.cuda())

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    for epoch in range(MAX_EPOCH):
        if epoch % 10 == 0:
            adjust_lr(optimizer, epoch, LR)
        epoch_loss_cls1, epoch_loss_cls2, epoch_loss_seg1, epoch_loss_seg2 = 0, 0, 0, 0
        epoch_loss_sum = 0
        for index, data in enumerate(train_loader):
            img, lbl = data['img'].cuda().float(), data['lbl'].cuda().float()
            img_cls, adj = data['cls'].cuda().float().squeeze(1), data['adj'].cuda().float()
            geo, grp_lbl = data['geo'].cuda().float(), data['grp_lbl'].cuda().float()
            # import pdb; pdb.set_trace()
            stage1_seg, stage1_cls, stage2_seg, stage2_cls = model(img, adj, geo, return_type='result')
            
            loss_cls1 = criterion_cls(stage1_cls, img_cls.long()) * 0.2
            loss_cls2 = criterion_cls(stage2_cls, img_cls.long()) * 0.2
            stage2_seg = torch.transpose(stage2_seg, 1, 2).contiguous()
            loss_seg2 = criterion_seg(stage2_seg, grp_lbl.long())
            loss_seg1 = dice_loss(stage1_seg, lbl) * 0.1
            loss_sum = loss_cls1 + loss_seg1 + loss_seg2 + loss_cls2
            
            epoch_loss_cls1 += float(loss_cls1)
            epoch_loss_cls2 += float(loss_cls2)
            epoch_loss_seg1 += float(loss_seg1)
            epoch_loss_seg2 += float(loss_seg2)
            epoch_loss_sum += float(loss_sum)
            
            accelerator.backward(loss_sum)
            optimizer.step()
            optimizer.zero_grad()

            if index % 10 == 0:
                ss = 'epcoh:' + str(epoch).zfill(4) + '/{}, '.format(MAX_EPOCH) + str(index).zfill(4)
                ss += ':cls1 loss:{:.4f}, cls2 loss:{:.4f}, seg1 loss:{:.4f}, '.format(float(loss_cls1), float(loss_cls2), float(loss_seg1))
                ss += 'seg2 loss:{:.4f}, sum loss:{:.4f}'.format(float(loss_seg2), float(loss_sum))
                print(ss)
                loss_f.write(ss + '\n')
                loss_f.flush()
        num_index = index + 1
        ss = '\ntotal epcoh:' + str(epoch).zfill(4) + '/{}, '.format(MAX_EPOCH)
        ss += ':cls1 loss:{:.4f}, cls2 loss:{:.4f}, seg1 loss:{:.4f}, '.format(float(epoch_loss_cls1) / num_index, float(epoch_loss_cls2) / num_index, float(epoch_loss_seg1) / num_index)
        ss += 'seg2 loss:{:.4f}, sum loss:{:.4f}'.format(float(epoch_loss_seg2) / num_index, float(epoch_loss_sum) / num_index)
        print(ss)
        loss_f.write(ss + '\n')
        loss_f.flush()
        if epoch == 1:
            torch.save(accelerator.unwrap_model(model).state_dict(), out_path + '{}.pth'.format(epoch))
        if (epoch + 1) % 200 == 0:
            torch.save(accelerator.unwrap_model(model).state_dict(), out_path + '{}.pth'.format(epoch + 1))
        if (epoch + 1) == MAX_EPOCH:
            torch.save(accelerator.unwrap_model(model).state_dict(), out_path + 'checkpoint.pth'.format(epoch + 1))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/pretrain_data/')
    parser.add_argument('--output', type=str, default='./output/pretrain/')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()
    print(opt)
    main(opt.data, opt.output)

