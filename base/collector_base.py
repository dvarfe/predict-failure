from abc import ABC, abstractmethod
import os

import pandas as pd


class AbstractDataCollector(ABC):
    """Базовый класс для всех сборщиков данных"""

    def __init__(self, config=None):
        self.update_config(config or {})
        # Имя файла для сохранения данных — по имени класса
        self._csv_path = f"storage/data/{self.__class__.__name__}.csv"
        os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)

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
