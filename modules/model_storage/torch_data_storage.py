import os
import json
from datetime import datetime
from typing import Any, Dict

import torch

from ..base.model_storage_base import AbstractModelStorage


class TorchDataStorage(AbstractModelStorage):

    def __init__(self, storage_dir: str = "storage/tensors"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _item_folder(self, name: str) -> str:
        return os.path.join(self.storage_dir, name)

    def _index_path(self) -> str:
        return os.path.join(self.storage_dir, "index.json")

    def _read_index(self) -> Dict[str, Any]:
        idx = self._index_path()
        if os.path.exists(idx):
            with open(idx, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _write_index(self, index: Dict[str, Any]):
        with open(self._index_path(), "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def save(self, name: str, obj: Any) -> str:
        base_name = name
        folder = self._item_folder(base_name)
        os.makedirs(folder, exist_ok=True)

        file_path = os.path.join(folder, f"{base_name}.pt")
        torch.save(obj, file_path)

        cfg = {
            "name": base_name,
            "ext": ".pt",
            "date": datetime.utcnow().isoformat() + "Z",
            "type": obj.__class__.__name__,
        }
        with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        index = self._read_index()
        index[base_name] = cfg
        self._write_index(index)

        return file_path

    def load(self, name: str) -> Any:
        base_name = name
        folder = self._item_folder(base_name)
        file_path = os.path.join(folder, f"{base_name}.pt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Saved tensor/object not found at '{file_path}'")
        return torch.load(file_path)

    def list_models(self) -> dict:
        return self._read_index()
