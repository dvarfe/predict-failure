import pandas as pd
from core.config import ConfigManager
from collectors import DICT_COLLECTORS

class SystemManager:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.data = None
        self.predictions = {}
        self.setup_config()

    def setup_config(self):
        self.collectors = {}
        self.models = {}

        DICT_COLLECT_FOR_OS = DICT_COLLECTORS.get(self.config_manager.get_system())
        for name, config in self.config_manager.get_collectors().items():
            self.collectors[name] = DICT_COLLECT_FOR_OS.get(name)(config)
            # self.register_collector(name, config)

    def find_objects(self):
        result = {}
        for name, collector in self.collectors.items():
            try:
                objects = collector.find_objects()
            except Exception:
                objects = []
            cfg = self.config_manager.get_collector_config(name)
            selected = cfg.get('selected_objects', [])
            result[name] = {"objects": objects, "selected": selected}
        return result

    # # --- Регистрация ---
    # def register_collector(self, name: str, config):
    #     self.collectors[name] = DICT_COLLECTORS.get(name)(config)

    def update_collectors(self):
        # обновить все конфиги после изменений
        for name, collector in self.collectors.items():
            cfg = self.config_manager.get_collector_config(name)
            collector.update_config(cfg)

    def register_model(self, name: str, model_cls):
        self.models[name] = model_cls

    # --- Работа с данными ---
    def collect_data(self, collector_name: str, objects=None):
        collector = self.collectors[collector_name]
        # If objects not provided, try to get selected objects from config
        if objects is None:
            cfg = self.config_manager.get_collector_config(collector_name)
            objects = cfg.get('selected_objects', None)
        self.data = collector.collect(objects=objects)
        return self.data

    # --- Работа с моделями ---
    def apply_model(self, model_name: str):
        if self.data is None:
            raise ValueError("Нет данных для применения модели")
        model = self.models[model_name]()
        model.fit(self.data)  # если модель обучаемая
        preds = model.predict(self.data)
        self.predictions[model_name] = preds
        return preds
