from .model_storage import ModelStorage
from .torch_data_storage import TorchDataStorage

DICT_MODEL_STORAGES = {
    "default": ModelStorage,
    "torch": TorchDataStorage,
}

DEFAULT_MODEL_STORAGE_TYPE = "default"
DEFAULT_MODEL_STORAGE_PARAMS = {
    "default": {
        "storage_dir": "storage/models",
    }
}
