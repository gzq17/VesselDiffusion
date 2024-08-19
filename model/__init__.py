from config.structured import ProjectConfig
from .model2DTo3D import Two2ThreeDiffusionModel
from .model_utils import set_requires_grad

def get_model(cfg: ProjectConfig):
    model = Two2ThreeDiffusionModel(**cfg.model)
    if cfg.run.freeze_feature_model:
        set_requires_grad(model.feature_model, False)
    return model

