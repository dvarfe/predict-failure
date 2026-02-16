"""
Общие типы и метаданные для всех коллекторов
"""

from enum import Enum
from typing import Dict


class FeatureType(Enum):
    """Тип признака для определения способа отображения"""
    NUMERICAL = "numerical"     # Числовой
    CATEGORICAL = "categorical"  # Категориальный
    TIMESTAMP = "time"     # Временная метка
    IDENTIFIER = "identifier"   # Идентификатор


class FeatureMetadata:
    """Метаданные признака"""

    def __init__(self, name: str, feature_type: FeatureType, unit: str = "",
                 description: str = ""):
        self.name = name
        self.type = feature_type
        self.unit = unit
        self.description = description

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "type": self.type.value,
            "unit": self.unit,
            "description": self.description,
        }
