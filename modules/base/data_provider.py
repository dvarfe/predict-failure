from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class AbstractDataProvider(ABC):

    def __init__(self, provider_type: str, device_name: Optional[str] = None):
        self.provider_type = provider_type
        self.device_name = device_name

    @abstractmethod
    def save_dataframe(self, name: str, df: pd.DataFrame, mode: str = "overwrite") -> str:
        raise NotImplementedError()

    @abstractmethod
    def load_dataframe(self, name: str, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Optional[pd.DataFrame]:
        raise NotImplementedError()

    @abstractmethod
    def remove(self, name: str) -> bool:
        raise NotImplementedError()
