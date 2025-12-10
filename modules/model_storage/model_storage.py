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
        dummy = ['DummyConstModel']
        index = self.read_index()
        if index:
            return list(index.keys()) + dummy
        return dummy

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
        if base_name == 'DummyConstModel':
            return DummyConstModel()
        elif base_name == 'DummyRandModel':
            return DummyRandModel()
        folder = self._model_folder(base_name)
        cfg = self._read_model_config(folder)
        ext = cfg.get('ext', '')
        if ext == '.onnx':
            return onnx.load(os.path.join(folder, f"{base_name}{ext}"))
        elif ext == '.pkl' or ext == '':
            return pickle.load(open(os.path.join(folder, f"{base_name}{ext}"), 'rb'))
        else:
            raise ValueError(f"Unknown model extension '{ext}' for model '{name}'")

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
