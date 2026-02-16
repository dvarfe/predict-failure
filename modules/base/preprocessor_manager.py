import os
import pandas as pd
from joblib import load


class PreprocessorManager:
    """Класс для управления препроцессорами моделей"""

    def __init__(self, preprocessors_dir: str = None):
        """
        Args:
            preprocessors_dir: Директория с препроцессорами (по умолчанию: storage/preprocessors)
        """
        self.preprocessors_dir = preprocessors_dir or os.path.join(os.getcwd(), 'storage', 'preprocessors')

    def load_preprocessor(self, preprocessor_name: str):
        """
        Загрузить препроцессор из .joblib файла

        Args:
            preprocessor_name: Имя препроцессора или путь к .joblib файлу

        Returns:
            Загруженный препроцессор или None, если не удалось загрузить
        """
        if not preprocessor_name:
            return None

        # Добавляем расширение, если его нет
        if not preprocessor_name.endswith('.joblib'):
            preprocessor_name += '.joblib'

        # Все препроцессоры лежат в папке storage/preprocessors
        preprocessor_path = os.path.join(self.preprocessors_dir, preprocessor_name)

        if os.path.exists(preprocessor_path):
            try:
                preprocessor = load(preprocessor_path)
                print(f"Препроцессор загружен из: {preprocessor_path}")
                return preprocessor
            except Exception as e:
                print(f"Ошибка при загрузке препроцессора {preprocessor_path}: {e}")
                return None

        print(f"Препроцессор {preprocessor_name} не найден в: {preprocessor_path}")
        return None

    def apply_preprocessing(self, data: pd.DataFrame, preprocessor_name: str) -> pd.DataFrame:
        """
        Применить препроцессинг к данным

        Args:
            data: Исходные данные
            preprocessor_name: Имя препроцессора

        Returns:
            Обработанные данные
        """
        preprocessor = self.load_preprocessor(preprocessor_name)

        if preprocessor is None:
            print(f"Препроцессор {preprocessor_name} не найден, возвращаем исходные данные")
            return data

        try:
            processed_data = preprocessor.transform(data)
            print(f"Препроцессинг {preprocessor_name} успешно применён")
            return processed_data
        except Exception as e:
            print(f"Ошибка при применении препроцессинга {preprocessor_name}: {e}")
            return data
