from typing import Dict, Any, List
from torch.utils.data import DataLoader
import pandas as pd
import json
import os
from datetime import datetime

from ..base.model_registry_base import AbstractModelRegistry
from ..models import DICT_MODELS, MODEL_PARAMETERS


class ModelRegistry(AbstractModelRegistry):
    """Реестр доступных моделей для обучения"""

    def __init__(self):
        self.models_dict = DICT_MODELS

    def list_models(self) -> List[str]:
        """Получить список доступных моделей"""
        return list(self.models_dict.keys())

    def get_model_class(self, model_name: str):
        """Получить класс модели по имени"""
        if model_name not in self.models_dict:
            raise ValueError(f"Модель '{model_name}' не найдена. Доступные модели: {list(self.models_dict.keys())}")
        return self.models_dict[model_name]

    def create_model(self, model_name: str, params: Dict[str, Any] = None):

        if params is None:
            params = {}

        model_class = self.get_model_class(model_name)

        model_params = self._filter_model_params(model_name, params)

        try:
            return model_class(**model_params)
        except TypeError as e:
            print(f"Предупреждение: не удалось создать модель с параметрами {model_params}: {e}")
            return model_class()

    def fit_model(self, model_name: str, dataloader: DataLoader, id_col: str, params: Dict[str, Any] = None):
        try:
            model = self.create_model(model_name, params)
            return model.fit(dataloader, id_col=id_col)
        except Exception as e:
            print(f"Ошибка при обучении модели {model_name}: {e}")
            return None

    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Получить описание параметров модели с их типами, значениями по умолчанию и описанием."""
        return MODEL_PARAMETERS.get(model_name, {})

    def _filter_model_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        model_params_desc = self.get_model_parameters(model_name)
        filtered_params = {}

        for param_name, param_value in params.items():
            if param_name in model_params_desc:
                param_desc = model_params_desc[param_name]
                param_type = param_desc.get('type', 'str')

                try:
                    if param_value == '' or param_value is None:
                        if param_desc.get('nullable', False):
                            filtered_params[param_name] = None
                        elif 'default' in param_desc:
                            filtered_params[param_name] = param_desc['default']
                        continue

                    if param_type == 'int':
                        filtered_params[param_name] = int(param_value)
                    elif param_type == 'float':
                        filtered_params[param_name] = float(param_value)
                    elif param_type == 'bool':
                        filtered_params[param_name] = bool(param_value)
                    else:
                        filtered_params[param_name] = param_value
                except (ValueError, TypeError):
                    if 'default' in param_desc:
                        filtered_params[param_name] = param_desc['default']

        return filtered_params

    def register_model(self, 
                      model_name: str, 
                      model_path: str, 
                      model_type: str, 
                      component_type: str = "drive", 
                      preprocessor: str = None,
                      index_path: str = "storage/models/index.json"):
        """
        Регистрирует обученную модель в index.json
        
        Args:
            model_name: Имя модели для регистрации
            model_path: Путь к файлу модели
            model_type: Тип модели (например, SurvPredictor, CoxTimeVaryingEstimator)
            component_type: Тип компонента (cpu, battery, gpu, drive)
            preprocessor: Имя препроцессора (опционально)
            index_path: Путь к файлу index.json
        """
        # Загружаем существующий index.json
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index_data = json.load(f)
        else:
            # Создаем новую структуру если файл не существует
            index_data = {
                "cpu": {},
                "battery": {},
                "gpu": {},
                "drive": {}
            }
        
        # Подготавливаем данные о модели
        model_info = {
            "name": model_name,
            "path": model_path,
            "created_at": datetime.now().isoformat(),
            "type": model_type
        }
        
        # Добавляем препроцессор если указан
        if preprocessor:
            model_info["preprocessor"] = preprocessor
        
        # Добавляем модель в соответствующую категорию
        if component_type not in index_data:
            index_data[component_type] = {}
        
        index_data[component_type][model_name] = model_info
        
        # Сохраняем обновленный index.json
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"Модель '{model_name}' зарегистрирована в {index_path}")