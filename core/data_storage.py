import os
from typing import Iterable, Optional
import pandas as pd

from base.data_provider import AbstractDataProvider


class FileSystemDataProvider(AbstractDataProvider):
    def __init__(self, provider_type: str = 'fs', base_dir: str = "storage/data", device_name: Optional[str] = None):
        self.base_dir = base_dir
        super().__init__(provider_type=provider_type, device_name=device_name)
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, name: str, ext: str = "csv") -> str:
        base = self.base_dir
        base = os.path.join(self.base_dir, str(self.device_name))
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f"{name}.{ext}")

    def save_dataframe(self, name: str, df: pd.DataFrame, mode: str = "overwrite") -> str:
        path = self._path(name, ext='csv')
        if mode == 'overwrite' or not os.path.exists(path):
            df.to_csv(path, index=False)
        elif mode == 'append':
            write_header = not os.path.exists(path) or os.path.getsize(path) == 0
            df.to_csv(path, mode='a', header=write_header, index=False)
        else:
            raise ValueError(f"Unknown mode '{mode}' for saving dataframe.")
        return path

    def load_dataframe(self, name: str, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Optional[pd.DataFrame]:
        path = self._path(name, ext='csv')
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        if (start_time is not None or end_time is not None) and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            if start_time is not None:
                df = df[df['timestamp'] >= float(start_time)]
            if end_time is not None:
                df = df[df['timestamp'] <= float(end_time)]
            df = df.reset_index(drop=True)
        return df

    def remove(self, name: str) -> bool:
        path = self._path(name, ext='csv')
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
