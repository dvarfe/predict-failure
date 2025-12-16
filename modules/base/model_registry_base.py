from abc import ABC, abstractmethod
from typing import Dict, Any, List
from torch.utils.data import DataLoader
import pandas as pd


class AbstractModelRegistry(ABC):
    """Абстрактный класс для регистра моделей"""

    @abstractmethod
    def list_models(self) -> List[str]:
        """Получить список доступных моделей"""
        pass

    @abstractmethod
    def fit_model(self, model_name: str, dataloader: DataLoader, params: Dict[str, Any] = None):
        """Обучить модель по имени"""
        pass

    @abstractmethod
    def get_model_class(self, model_name: str):
        """Получить класс модели по имени"""
        pass

    @abstractmethod
    def create_model(self, model_name: str, params: Dict[str, Any] = None):
        """Создать экземпляр модели"""
        pass