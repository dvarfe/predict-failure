from abc import ABC, abstractmethod
import pandas as pd


class AbstractDataCollector(ABC):
    """Базовый класс для всех сборщиков данных"""

    def __init__(self, config=None):
        self.update_config(config or {})
        self.data_provider = None

    def set_data_provider(self, provider):
        self.data_provider = provider

    def get_history(self, start_time: float = None, end_time: float = None):
        name = self.__class__.__name__
        provider = self.data_provider
        if provider is None:
            raise RuntimeError("No data_provider!")

        df = provider.load_dataframe(name, start_time=start_time, end_time=end_time)
        return df if df is not None else pd.DataFrame()

    def save_dataframe(self, df: pd.DataFrame, mode: str = "append") -> str:
        name = self.__class__.__name__
        provider = self.data_provider
        if provider is None:
            raise RuntimeError("No data_provider!")

        return provider.save_dataframe(name, df, mode=mode)

    def get_history(self, start_time: float = None, end_time: float = None):
        name = self.__class__.__name__
        provider = self.data_provider
        if provider is None:
            raise RuntimeError("No data_provider!")

        df = provider.load_dataframe(name, start_time=start_time, end_time=end_time)
        return df if df is not None else pd.DataFrame()

    def update_config(self, config):
        self.interval = config.get("interval", 1)  # сек между замерами

    @abstractmethod
    def find_objects(self):
        """Найти доступные объекты для мониторинга (например, устройства или клиентов)"""
        pass

    @abstractmethod
    def collect(self, objects=None) -> pd.DataFrame:
        """Собрать данные по выбранным объектам"""
        pass
