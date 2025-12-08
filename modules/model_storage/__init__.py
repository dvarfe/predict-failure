from .model_storage import ModelStorage, PickleModelStorage, ONNXModelStorage

DICT_MODEL_STORAGES = {
    "default": ModelStorage,
    "pickle": PickleModelStorage,
    "onnx": ONNXModelStorage,
}

DEFAULT_MODEL_STORAGE_TYPE = "default"
DEFAULT_MODEL_STORAGE_PARAMS = {
    "default": {
        "storage_dir": "storage/models",
    },
    "pickle": {
        "storage_dir": "storage/models",
    },
    "onnx": {
        "storage_dir": "storage/models", 
    }
}