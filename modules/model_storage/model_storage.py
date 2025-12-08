import pickle
import os

import onnx

from ..base.model_storage_base import AbstractModelStorage

class PickleModelStorage(AbstractModelStorage):
    def __init__(self, storage_dir: str = "storage/models"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.storage_dir, f"{name}.pkl")

    def list_models(self):
        return [model for model in os.listdir(self.storage_dir) if model.endswith('.pkl')]


    def save(self, name: str, model) -> str:
        path = self._path(name)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        return path

    def load(self, name: str):
        path = self._path(name)
        with open(path, 'rb') as f:
            return pickle.load(f)

class ONNXModelStorage(AbstractModelStorage):
    def __init__(self, storage_dir: str = "storage/models"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.storage_dir, f"{name}.onnx")

    def list_models(self):
        return [model for model in os.listdir(self.storage_dir) if model.endswith('.onnx')]

    def save(self, name: str, model) -> str:
        path = self._path(name)
        with open(path, 'wb') as f:
            f.write(model.SerializeToString())
        return path

    def load(self, name: str):
        path = self._path(name)
        return onnx.load(path)

class ModelStorage(AbstractModelStorage):   
    
    def __init__(self, storage_dir: str = "storage/models"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.pickle_storage = PickleModelStorage(storage_dir=storage_dir)
        self.onnx_storage = ONNXModelStorage(storage_dir=storage_dir)

    def list_models(self):
        return os.listdir(self.storage_dir)
    
    def load(self, name):
        if name.endswith('.onnx'):
            base_name = name[:-5] if name.endswith('.onnx') else name
            return self.onnx_storage.load(base_name)
        elif name.endswith('.pkl'):   
            base_name = name[:-4] if name.endswith('.pkl') else name
            return self.pickle_storage.load(base_name)
        else:
            raise ValueError(f"Unknown model format for '{name}'")
    
    def save(self, name: str, model) -> str:
        if name.endswith('.onnx'):
            base_name = name[:-5] if name.endswith('.onnx') else name
            return self.onnx_storage.save(base_name, model)
        elif name.endswith('.pkl'):
            base_name = name[:-4] if name.endswith('.pkl') else name
            return self.pickle_storage.save(base_name, model)
        else:
            raise ValueError(f"Unknown model format for '{name}'")

