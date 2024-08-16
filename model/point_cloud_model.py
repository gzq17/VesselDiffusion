import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from torch import Tensor

from .pvcnn.pvcnn import PVCNN2

class PointCloudModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 64,
        dropout: float = 0.1,
        width_multiplier: int = 1,
        voxel_resolution_multiplier: int = 1,
    ):
        super().__init__()
        self.autocast_context = torch.autocast('cuda', dtype=torch.float32)
        self.model = PVCNN2(
            embed_dim=embed_dim,
            num_classes=out_channels,
            extra_feature_channels=(in_channels - 3),
            dropout=dropout, width_multiplier=width_multiplier, 
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.model.classifier[-1].bias.data.normal_(0, 1e-6)
        self.model.classifier[-1].weight.data.normal_(0, 1e-6)

    def forward(self, inputs: Tensor, t: Tensor) -> Tensor:
        with self.autocast_context:
            return self.model(inputs.transpose(1, 2), t).transpose(1, 2)
