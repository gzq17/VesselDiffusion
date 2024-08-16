from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from hydra.conf import RunDir


@dataclass
class CustomHydraRunDir(RunDir):
    dir: str = './output/VesselDiffusion_model/${now:%Y-%m-%d--%H-%M-%S}'

@dataclass
class RunConfig:
    name: str = 'run'
    job: str = 'train' ##sample  train
    mixed_precision: str = 'fp16'  # 'no'
    cpu: bool = False
    seed: int = 42
    val_before_training: bool = False
    vis_before_training: bool = False
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    max_steps: int = 200_000
    checkpoint_freq: int = 20_000
    val_freq: int = 5_000
    vis_freq: int = 10_000
    log_step_freq: int = 20
    print_step_freq: int = 100

    # Inference config
    num_inference_steps: int = 1000
    diffusion_scheduler: Optional[str] = 'ddpm'
    num_samples: int = 1
    num_sample_batches: Optional[int] = None
    sample_from_ema: bool = False 
    sample_save_evolutions: bool = True  # temporarily set by default
    
    # Training config
    freeze_feature_model: bool = True

    # Coloring training config
    coloring_training_noise_std: float = 0.0
    coloring_sample_dir: Optional[str] = None


@dataclass
class LoggingConfig:
    wandb: bool = False##改
    wandb_project: str = 'VesselDiffusion'


@dataclass
class PointCloudProjectionModelConfig:
    
    # Feature extraction arguments
    image_size: int = '${dataset.image_size}'
    use_local_colors: bool = False
    use_gcn_features: bool = True
    use_local_features: bool = True
    use_global_features: bool = True###改
    color_channels: int = 3
    pre_train_model: str = ''


@dataclass
class PointCloudDiffusionModelConfig(PointCloudProjectionModelConfig):

    # Diffusion arguments
    beta_start: float = 1e-5  # 0.00085
    beta_end: float = 8e-3  # 0.012
    beta_schedule: str = 'linear'  # 'custom'

    # Point cloud model arguments
    point_cloud_model_embed_dim: int = 64


@dataclass
class PointCloudColoringModelConfig(PointCloudProjectionModelConfig):

    # Projection arguments
    predict_shape: bool = False
    predict_color: bool = True

    # Point cloud model arguments
    point_cloud_model_layers: int = 1
    point_cloud_model_embed_dim: int = 64


@dataclass
class DatasetConfig:
    type: str


@dataclass
class PointCloudDatasetConfig(DatasetConfig):
    eval_split: str = 'val'
    max_points: int = 16_384
    image_size: int = 336
    scale_factor: float = 1.0
    restrict_model_ids: Optional[List] = None  # for only running on a subset of data points


@dataclass
class CO3DConfig(PointCloudDatasetConfig):
    type: str = 'coronary'
    # root: str = os.getenv('CO3DV2_DATASET_ROOT')
    root: str = ''
    test_origin: str = 'real'
    category: str = 'hydrant'
    subset_name: str = 'fewview_dev'
    mask_images: bool = '${model.use_mask}'


@dataclass
class AugmentationConfig:
    pass


@dataclass
class DataloaderConfig:
    batch_size: int = 16 
    num_workers: int = 6 


@dataclass
class LossConfig:
    diffusion_weight: float = 1.0
    rgb_weight: float = 1.0
    consistency_weight: float = 1.0


@dataclass
class CheckpointConfig:
    resume: Optional[str] = None
    resume_training: bool = True
    resume_training_optimizer: bool = True
    resume_training_scheduler: bool = True
    resume_training_state: bool = True


@dataclass
class ExponentialMovingAverageConfig:
    use_ema: bool = False
    decay: float = 0.999
    update_every: int = 20


@dataclass
class OptimizerConfig:
    type: str
    name: str
    lr: float = 1e-3
    weight_decay: float = 0.0
    scale_learning_rate_with_batch_size: bool = False
    gradient_accumulation_steps: int = 1
    clip_grad_norm: Optional[float] = 50.0  # 5.0
    kwargs: Dict = field(default_factory=lambda: dict())


@dataclass
class AdadeltaOptimizerConfig(OptimizerConfig):
    type: str = 'torch'
    name: str = 'Adadelta'
    kwargs: Dict = field(default_factory=lambda: dict(
        weight_decay=1e-6,
    ))


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    type: str = 'torch'
    name: str = 'AdamW'
    weight_decay: float = 1e-6
    kwargs: Dict = field(default_factory=lambda: dict(betas=(0.95, 0.999)))


@dataclass
class SchedulerConfig:
    type: str
    kwargs: Dict = field(default_factory=lambda: dict())


@dataclass
class LinearSchedulerConfig(SchedulerConfig):
    type: str = 'transformers'
    kwargs: Dict = field(default_factory=lambda: dict(
        name='linear',
        num_warmup_steps=0,
        num_training_steps="${run.max_steps}",
    ))


@dataclass
class CosineSchedulerConfig(SchedulerConfig):
    type: str = 'transformers'
    kwargs: Dict = field(default_factory=lambda: dict(
        name='cosine',
        num_warmup_steps=2000,  # 0
        num_training_steps="${run.max_steps}",
    ))


@dataclass
class ProjectConfig:
    run: RunConfig
    logging: LoggingConfig
    dataset: PointCloudDatasetConfig
    augmentations: AugmentationConfig
    dataloader: DataloaderConfig
    loss: LossConfig
    model: PointCloudProjectionModelConfig
    ema: ExponentialMovingAverageConfig
    checkpoint: CheckpointConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    defaults: List[Any] = field(default_factory=lambda: [
        'custom_hydra_run_dir',
        {'run': 'default'},
        {'logging': 'default'},
        {'model': 'diffrec'},
        {'dataset': 'co3d'},
        {'augmentations': 'default'},
        {'dataloader': 'default'},
        {'ema': 'default'},
        {'loss': 'default'},
        {'checkpoint': 'default'},
        {'optimizer': 'adam'},
        {'scheduler': 'cosine'},
    ])


cs = ConfigStore.instance()
cs.store(name='custom_hydra_run_dir', node=CustomHydraRunDir, package="hydra.run")
cs.store(group='run', name='default', node=RunConfig)
cs.store(group='logging', name='default', node=LoggingConfig)
cs.store(group='model', name='diffrec', node=PointCloudDiffusionModelConfig)
cs.store(group='model', name='coloring_model', node=PointCloudColoringModelConfig)
cs.store(group='dataset', name='co3d', node=CO3DConfig)
# TODO
# cs.store(group='dataset', name='shapenet_r2n2', node=ShapeNetR2N2Config)
# cs.store(group='dataset', name='shapenet_nmr', node=ShapeNetNMRConfig)
cs.store(group='augmentations', name='default', node=AugmentationConfig)
cs.store(group='dataloader', name='default', node=DataloaderConfig)
cs.store(group='loss', name='default', node=LossConfig)
cs.store(group='ema', name='default', node=ExponentialMovingAverageConfig)
cs.store(group='checkpoint', name='default', node=CheckpointConfig)
cs.store(group='optimizer', name='adadelta', node=AdadeltaOptimizerConfig)
cs.store(group='optimizer', name='adam', node=AdamOptimizerConfig)
cs.store(group='scheduler', name='linear', node=LinearSchedulerConfig)
cs.store(group='scheduler', name='cosine', node=CosineSchedulerConfig)
cs.store(name='config', node=ProjectConfig)
