import pickle
import os
import json
from datetime import datetime

import onnx

from ..base.model_storage_base import AbstractModelStorage
from ..models.dummy_const_model import DummyConstModel  # TODO: убрать всё, что с этим связано
from ..models.dummy_rand_model import DummyRandModel


class ModelStorage(AbstractModelStorage):

    def __init__(self, storage_dir: str = "storage/models"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _model_folder(self, base_name: str) -> str:
        return os.path.join(self.storage_dir, base_name)

    def list_models(self):
        return self.read_index()

    def _index_path(self) -> str:
        return os.path.join(self.storage_dir, 'index.json')

    def read_index(self) -> dict:
        idx = self._index_path()
        if os.path.exists(idx):
            with open(idx, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _write_index(self, index: dict):
        idx = self._index_path()
        with open(idx, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _read_model_config(self, folder: str) -> dict:
        cfg_path = os.path.join(folder, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def load(self, name: str):
        base_name = name
        if base_name.endswith('DummyConstModel'):
            return DummyConstModel()
        elif base_name.endswith('DummyRandModel'):
            return DummyRandModel()

        if '/' not in base_name:
            raise ValueError(f"Model name must be in 'device/model.ext' format, got '{name}'")

        full_path = os.path.join(self.storage_dir, base_name)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Model file for '{name}' not found at '{full_path}'")

        _, ext = os.path.splitext(full_path)
        if ext.lower() == '.onnx':
            return onnx.load(full_path)
        else:
            return pickle.load(open(full_path, 'rb'))

    def save(self, name: str, model) -> str:
        base_name = name
        ext = ''
        if name.endswith('.onnx'):
            base_name = name[:-5]
            ext = '.onnx'
        elif name.endswith('.pkl'):
            base_name = name[:-4]
            ext = '.pkl'

        folder = self._model_folder(base_name)
        os.makedirs(folder, exist_ok=True)

        # choose storage based on ext
        if ext == '.onnx':
            onnx.save(model, os.path.join(folder, f"{base_name}{ext}"))
            mtype = 'onnx'
            file_ext = '.onnx'
        else:
            # default to pickle
            with open(os.path.join(folder, f"{base_name}{ext}"), 'wb') as f:
                pickle.dump(model, f)
            mtype = 'pickle'
            file_ext = '.pkl'

        cfg = {
            'name': base_name,
            'type': mtype,
            'ext': file_ext,
            'description': '',
            'date': datetime.isoformat() + 'Z'
        }
        cfg_path = os.path.join(folder, 'config.json')
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        index = self._read_index()
        index[base_name] = cfg
        self._write_index(index)

        return os.path.join(folder, f"{base_name}{file_ext}")

    def list_datasets(self) -> dict:
        result = {}
        index = self.read_index()

        return result

    def register_model(self, device: str, model_path: str, meta: dict = None) -> None:
        if meta is None:
            meta = {}

        index = self.read_index()
        if not isinstance(index, dict):
            index = {}

        if device not in index or not isinstance(index[device], dict):
            index[device] = {}
        # store the model entry under the device using the provided model_path
        index[device][model_path] = meta or {}

        self._write_index(index)
