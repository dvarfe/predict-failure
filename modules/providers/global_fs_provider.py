import os
from typing import Optional, Dict, List
import pandas as pd
from torch.utils.data import DataLoader

from ..base.data_provider import AbstractDataProvider
from .dataloader import DiskDataset


class GlobalFileSystemProvider(AbstractDataProvider):

    def __init__(self, provider_type: str = 'global_fs', base_dir: str = 'storage/data'):
        super().__init__(provider_type=provider_type, device_name=None)
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        pass

    def list_devices(self) -> List[str]:
        return [name for name in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, name))]

    def list_datasets(self, device: Optional[str] = None) -> Dict[str, List[str]]:
        if device:
            dpath = os.path.join(self.base_dir, device)
            if not os.path.isdir(dpath):
                return {device: []}
            return {device: [os.path.splitext(f)[0] for f in os.listdir(dpath) if f.endswith('.csv')]}

        result = {}
        for dev in self.list_devices():
            result[dev] = [os.path.splitext(f)[0] for f in os.listdir(
                os.path.join(self.base_dir, dev)) if f.endswith('.csv')]
        return result

    def _dataset_path(self, device: str, name: str, ext: str = 'csv') -> str:
        return os.path.join(self.base_dir, device, f"{name}.{ext}")

    def save_dataframe(self, name: str, df: pd.DataFrame, device: str, mode: str = "overwrite") -> str:
        path = self._dataset_path(device, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if mode == 'overwrite' or not os.path.exists(path):
            df.to_csv(path, index=False)
        elif mode == 'append':
            write_header = not os.path.exists(path) or os.path.getsize(path) == 0
            df.to_csv(path, mode='a', header=write_header, index=False)
        return path

    def load_dataframe(self, name: str, start_time: Optional[float] = None, end_time: Optional[float] = None, device: Optional[str] = None) -> Optional[pd.DataFrame]:
        if device:
            # При вызове без имени набора данных, загружаем последний по дате изменения
            if not name:
                dpath = os.path.join(self.base_dir, device)
                csvs = [os.path.join(dpath, f) for f in os.listdir(dpath) if f.endswith('.csv')]
                path = max(csvs, key=os.path.getmtime)
            else:
                path = self._dataset_path(device, name)
        else:
            path = name

        df = pd.read_csv(path)
        if (start_time is not None or end_time is not None) and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            if start_time is not None:
                df = df[df['timestamp'] >= float(start_time)]
            if end_time is not None:
                df = df[df['timestamp'] <= float(end_time)]
            df = df.reset_index(drop=True)
        return df

    def get_dataloader(self, name: str, device: Optional[str] = None, batch_size: int = 32,
                       start_time: Optional[float] = None, end_time: Optional[float] = None,
                       ids: Optional[list] = None, id_col: Optional[str] = None,
                       mode: str = 'train') -> Optional[DataLoader]:
        """Получить DataLoader для указанного датасета"""

        if device:
            file_path = self._dataset_path(device, name)
        else:
            file_path = name

        if not os.path.exists(file_path):
            return None

        dataset = DiskDataset(
            mode=mode,
            file_paths=[file_path],
            shuffle_files=False,
            ids=ids,
            id_col=id_col,
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def remove(self, name: str, device: Optional[str] = None):
        if device is not None:
            path = self._dataset_path(device, name)
        else:
            path = name
        if os.path.exists(path):
            os.remove(path)
