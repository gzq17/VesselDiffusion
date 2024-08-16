from typing import Optional, Union
import torch
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers import ModelMixin
from torch import Tensor
import torch.nn.functional as F
from pretrain_DSFE import DualStreamFE
import copy
import os

SchedulerClass = Union[DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]


class PointCloudProjectionModel(ModelMixin):
    def __init__(
        self,
        image_size: int,
        use_local_colors: bool = False,
        use_local_features: bool = True,
        use_gcn_features: bool = True,
        use_global_features: bool = True,
        image_color_channels: int = 3,  # for the input image, not the points
        color_channels: int = 3,  # for the points, not the input image
        pre_train_model: str = '',
    ):
        super().__init__()
        self.image_size = image_size
        self.use_local_colors = use_local_colors
        self.use_local_features = use_local_features
        self.use_global_features = use_global_features
        self.use_gcn_features = use_gcn_features
        self.image_color_channels = image_color_channels
        self.color_channels = color_channels
        self.use_local_conditioning = self.use_local_colors or self.use_local_features
        self.use_global_conditioning = self.use_global_features
        # Create feature model
        self.feature_model = DualStreamFE()
        if os.path.exists(pre_train_model):
            self.feature_model.load_state_dict(torch.load(pre_train_model, map_location='cpu'))

        # Input size
        self.in_channels = 3  # 3 for 3D point positions
        if self.use_local_colors:
            self.in_channels += self.image_color_channels
        feature_dim_C = self.feature_model.feature_dim
        add_feature_dim_C = feature_dim_C
        if self.use_gcn_features:
            add_feature_dim_C = feature_dim_C * 2
        if self.use_local_features:
            self.in_channels += add_feature_dim_C
        if self.use_global_features:
            self.in_channels += add_feature_dim_C
        self.out_channels = 3
    
    def get_global_feature(self, cls_head_fea, gcn_cls_head):
        global_conditioning = []
        if self.use_global_features:
            global_conditioning.append(cls_head_fea)
            if self.use_gcn_features:
                global_conditioning.append(gcn_cls_head)
        global_conditioning = torch.cat(global_conditioning, dim=1)  # (B, D_cond)
        return global_conditioning
    
    def get_local_feature(self, image_rgb, deep_feature_one, gcn_fea):
        local_conditioning = []
        if self.use_local_colors:
            local_conditioning.append(image_rgb)
        if self.use_local_features:
            local_conditioning.append(deep_feature_one)
            if self.use_gcn_features:
                local_conditioning.append(gcn_fea)
        local_conditioning = torch.cat(local_conditioning, dim=1)  # (B, D_cond, H, W)
        return local_conditioning

    def sample_features(self, feature_map, points):
        N, C, H, W = feature_map.size()
        M = points.size(1)
        points = copy.deepcopy(points).float()
        points[:, :, [0, 1]] = points[:, :, [1, 0]]
        points[..., 0] = (points[..., 0] / (W - 1)) * 2 - 1
        points[..., 1] = (points[..., 1] / (H - 1)) * 2 - 1
        points = points.unsqueeze(2)
        sampled_features = F.grid_sample(feature_map, points, mode='nearest', padding_mode='zeros', align_corners=True)
        
        sampled_features = sampled_features.squeeze(3).permute(0, 2, 1)
        
        return sampled_features

    def projection_head(self, points, mean, std, hf_size, local_features, mip_img=None):
        index_p = (points * std + mean) * hf_size + hf_size
        local_features_proj = self.sample_features(local_features, index_p[:, :, 1:3])
        return local_features_proj

    def extend_feature(self, x_t, mip_img, mean, std, hf_size, t: Optional[Tensor], adj=None, geo=None):
        B, N = x_t.shape[:2]
        x_t_input = [x_t]
        # cls_head_fea, deep_feature_one, gcn_fea = self.feature_model(mip_img, adj, geo, return_type='feature')
        cnn_cls_fea, cnn_seg_fea, gcn_cls_fea, gcn_seg_fea = self.feature_model(mip_img, adj, geo, return_type='feature')
        # Local conditioning
        # import pdb;pdb.set_trace()
        if self.use_local_conditioning:
            local_features = self.get_local_feature(mip_img, cnn_seg_fea, gcn_seg_fea)
            if local_features.shape[-2:] != mip_img.shape[-2:]:
                raise ValueError(f'{local_features.shape=} and {mip_img.shape=}')
            local_features_proj = self.projection_head(x_t[:, :, :3], mean, std, hf_size, local_features=local_features)  # (B, N, D_local)

            x_t_input.append(local_features_proj)

        # Global conditioning
        if self.use_global_conditioning:
            global_features = self.get_global_feature(cnn_cls_fea, gcn_cls_fea)
            global_features = global_features.unsqueeze(1).expand(-1, N, -1)  # (B, D_global, N)

            x_t_input.append(global_features)

        # Concatenate together all the pointwise features
        x_t_input = torch.cat(x_t_input, dim=2)  # (B, N, D)

        return x_t_input

    def forward(self, batch, mode='train', **kwargs):
        """ The forward method may be defined differently for different models. """
        raise NotImplementedError()
