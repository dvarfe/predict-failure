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

        if not os.path.isabs(name):
            if name.startswith('storage/'):
                model_path = name
            else:
                model_path = os.path.join(self.storage_dir, name)
        else:
            model_path = name

        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        _, ext = os.path.splitext(model_path)
        if ext.lower() == '.onnx':
            return onnx.load(model_path)
        else:
            return pickle.load(open(model_path, 'rb'))

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

    def save_model(self, model, device: str, model_name: str) -> str:
        """Сохранить модель для конкретного устройства"""
        # Создаём папку для устройства
        device_folder = os.path.join(self.storage_dir, device)
        os.makedirs(device_folder, exist_ok=True)

        # Путь для модели
        model_path = os.path.join(device_folder, f"{model_name}.pkl")

        # Сохраняем модель
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Обновляем индекс — теперь индекс хранит модели в виде словаря { device: { model_name: model_info } }
        index = self.read_index()
        if device not in index or not isinstance(index[device], dict):
            index[device] = {}

        model_info = {
            'name': model_name,
            'path': model_path,
            'created_at': datetime.now().isoformat(),
            'type': type(model).__name__
        }

        # Сохраняем/перезаписываем модель по ключу model_name
        index[device][model_name] = model_info

        self._write_index(index)
        return model_path

    def delete_model(self, device: str, model_name: str) -> bool:
        """Удалить модель для конкретного устройства"""
        try:
            index = self.read_index()
            if device not in index:
                return False
            # Поддерживаем структуру, где index[device] — либо список (legacy), либо dict
            if isinstance(index[device], list):
                # legacy: список словарей
                model_to_delete = None
                for i, model_info in enumerate(index[device]):
                    if model_info.get('name') == model_name:
                        model_to_delete = i
                        break
                if model_to_delete is None:
                    return False
                model_info = index[device][model_to_delete]
                if 'path' in model_info and os.path.exists(model_info['path']):
                    os.remove(model_info['path'])
                del index[device][model_to_delete]
                if len(index[device]) == 0:
                    device_folder = os.path.join(self.storage_dir, device)
                    if os.path.exists(device_folder):
                        try:
                            os.rmdir(device_folder)
                        except OSError:
                            pass
                    del index[device]
            else:
                # Современный формат: словарь model_name -> model_info
                if model_name not in index[device]:
                    return False
                model_info = index[device][model_name]
                if 'path' in model_info and os.path.exists(model_info['path']):
                    os.remove(model_info['path'])
                del index[device][model_name]
                if len(index[device]) == 0:
                    device_folder = os.path.join(self.storage_dir, device)
                    if os.path.exists(device_folder):
                        try:
                            os.rmdir(device_folder)
                        except OSError:
                            pass
                    del index[device]

            self._write_index(index)
            return True
        except Exception as e:
            print(f"Ошибка при удалении модели {device}/{model_name}: {e}")
            return False
