from abc import ABC, abstractmethod
import pickle
import pandas as pd
from typing import Optional


class AbstractModel(ABC):
    """Базовый класс для ML-моделей (в т.ч. survival)"""

    def fit(self, data: pd.DataFrame, event_col: str = 'event', duration_col: str = 'duration'):
        """Обучить модель (если нужно)"""
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, times=None, id_col: Optional[str] = None) -> pd.DataFrame:
        """Сделать прогноз (например, функция выживания, риск).

        Parameters:
        - data: входной DataFrame с признаками
        - times: массив временных точек для функции выживания (может быть None)
        - id_col: имя столбца идентификатора в `data`, которое модель должна вернуть

        Возвращает DataFrame, где колонки — это временные метки (строки),
        а также колонка с идентификаторами, если `id_col` указан или модель добавляет 'id'.
        """
        pass

    def save(self, filepath: str):
        """Сохранить модель в файл"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
