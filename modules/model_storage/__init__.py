from .model_storage import ModelStorage
DICT_MODEL_STORAGES = {
    "default": ModelStorage,
}

DEFAULT_MODEL_STORAGE_TYPE = "default"
DEFAULT_MODEL_STORAGE_PARAMS = {
    "default": {
        "storage_dir": "storage/models",
    }
}
