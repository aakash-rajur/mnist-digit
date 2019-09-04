from src.models.keras.model import KerasModel
from src.pkg.config_constants import \
    MODEL_SAVE, \
    MODEL_CONFIG, \
    MODEL_TYPE, \
    MODEL_KERAS
from src.pkg.ensure_dir import ensure_dir


def construct_model_template() -> dict:
    model_template = {
        MODEL_KERAS: KerasModel
    }
    return model_template


def build_models(model_template: dict, base_dir: str, model_configs: dict) -> dict:
    model_map = {}
    for character, model_meta in model_configs.items():
        model_dir = ensure_dir(base_dir, model_meta[MODEL_SAVE])
        model_config = model_meta[MODEL_CONFIG]
        model_type = model_config[MODEL_TYPE]
        # noinspection PyPep8Naming
        Class = model_template[model_type]
        model_map[character] = Class(model_dir=model_dir, config=model_config)
    return model_map
