import pandas as pd
from core.config import ConfigManager
from modules.providers import DICT_DATA_PROVIDERS, DEFAULT_PROVIDER_PARAMS
from modules.providers.global_fs_provider import GlobalFileSystemProvider  # TODO: Избавиться от захардкоженного провайдера
from modules.schedulers import DICT_SCHEDULERS, DEFAULT_SCHEDULER_PARAMS
from modules.collectors import DICT_COLLECTORS
from modules.model_storage import ModelStorage, DICT_MODEL_STORAGES
from typing import Optional
from modules.base.feature_metadata import FeatureType


class SystemManager:
    def __init__(self, app=None):
        self.config_manager = ConfigManager()
        self.predictions = {}
        self._id_col_cache = {}
        self.fs_provider = GlobalFileSystemProvider()
        self._last_dataset_key = None
        self._last_dataset_df = None

        self.global_model_storage = ModelStorage()
        self.setup_config(app)

    def setup_collectors(self):
        self.collectors = {}
        self._id_col_cache = {}
        self._last_dataset_key = None
        self._last_dataset_df = None
        DICT_COLLECT_FOR_OS = DICT_COLLECTORS.get(self.config_manager.get_system())

        for name, config in self.config_manager.get_collectors().items():
            collector_obj = DICT_COLLECT_FOR_OS.get(name)(config)

            provider = self.create_data_provider(name, config)
            collector_obj.set_data_provider(provider)
            self.collectors[name] = collector_obj

    def create_data_provider(self, device_name: str, config: dict):
        global_cfg = self.config_manager.get_config().get(
            'default_data_provider', DEFAULT_PROVIDER_PARAMS)
        global_type = list(global_cfg.keys())[0]

        p_cfg = config.get('data_provider', global_cfg)
        p_type = list(p_cfg.keys())[0] if p_cfg else global_type
        p_cfg = {**p_cfg[p_type], **{'device_name': device_name}}

        Provider = DICT_DATA_PROVIDERS.get(p_type)
        return Provider(**p_cfg)

    def setup_scheduler(self, app=None):
        sched_cfg = self.config_manager.get_scheduler()
        name = sched_cfg.get('name', sched_cfg["name"])
        if name in DICT_SCHEDULERS:
            self.scheduler = DICT_SCHEDULERS[name](app, **sched_cfg)
        else:
            self.scheduler = DICT_SCHEDULERS[DEFAULT_SCHEDULER_PARAMS["name"]](app, **DEFAULT_SCHEDULER_PARAMS)
        self.scheduler_name = name

        self.scheduler.start()

    def setup_config(self, app=None):
        self.models = {}
        self.setup_collectors()
        self.setup_scheduler(app)

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

    def update_collectors(self):
        # обновить все конфиги после изменений
        self._id_col_cache = {}
        self._last_dataset_key = None
        self._last_dataset_df = None
        for name, collector in self.collectors.items():
            cfg = self.config_manager.get_collector_config(name)
            collector.update_config(cfg)

    def set_schedule(self, enabled=None, interval_value=None, interval_unit=None, selected_collectors=None):
        # Важно: эта функция только обновляет конфиг, но не применяет его к планировщику
        # К планировщику изменения применятюся через apply_schedule, когда пользователь нажмёт на сбор данных
        sched = self.config_manager.get_scheduler()
        if enabled is not None:
            sched['enabled'] = bool(enabled)
        if interval_value is not None:
            sched['interval_value'] = int(interval_value)
        if interval_unit is not None:
            sched['interval_unit'] = str(interval_unit)
        if selected_collectors is not None:
            sched['selected_collectors'] = list(selected_collectors)
        self.config_manager.update_scheduler_config(sched)

    def get_schedule(self):
        return self.config_manager.get_scheduler()

    def apply_schedule(self, schedule_config):
        if self.scheduler is None:
            return
        self.scheduler.apply_schedule(schedule_config, self.run_scheduled)

    def run_scheduled(self, selected_collectors):
        for collector_name in selected_collectors:
            if collector_name in self.collectors:
                self.collect_data(collector_name)
                print(f"Данные собраны с {collector_name}")

    # --- Работа с данными ---
    def collect_data(self, collector_name: str, objects=None):
        collector = self.collectors[collector_name]
        # If objects not provided, try to get selected objects from config
        if objects is None:
            cfg = self.config_manager.get_collector_config(collector_name)
            objects = cfg.get('selected_objects', None)
        data = collector.collect(objects=objects)
        return data

    # --- Работа с моделями ---
    def apply_model(self, model_name: str, data: pd.DataFrame, id_col: Optional[str] = None):
        if data is None:
            raise ValueError("Нет данных для применения модели")
        model = self.global_model_storage.load(model_name)

        preds = model.predict(data, id_col=id_col)

        preds = pd.DataFrame(preds)

        self.predictions[model_name] = preds
        return preds

    def save_model(self, model_name: str, model) -> str:
        return self.global_model_storage.save(model_name, model)

    def load_model(self, device_name: str, model_name: str):
        if device_name and device_name != 'general' and '/' not in model_name:
            path = f"{device_name}/{model_name}"
        else:
            path = model_name
        return self.global_model_storage.load(path)

    def list_models(self) -> list:
        return self.global_model_storage.list_models()

    def get_models(self) -> dict:
        return self.global_model_storage.list_models()

    def get_id_col(self, collector_name: str) -> Optional[str]:
        if collector_name in self._id_col_cache:
            return self._id_col_cache.get(collector_name)

        if collector_name not in self.collectors:
            return None
        collector = self.collectors[collector_name]

        metadata = collector.get_feature_metadata()
        for name, meta in metadata.items():
            if meta.type == FeatureType.IDENTIFIER:
                self._id_col_cache[collector_name] = name
                return name
        self._id_col_cache[collector_name] = None
        return None

    def predict_survival(self, device_name: str, model_name: str, data: pd.DataFrame) -> pd.Series:
        model = self.load_model(device_name, model_name)
        id_col = self.get_id_col(device_name)
        return model.predict(data, id_col=id_col)

    def list_datasets(self, device: str = None):
        return self.fs_provider.list_datasets(device=device)

    def load_dataframe(self, name: str, device: str = None, start_time: float = None, end_time: float = None):
        key = (name, device)
        if self._last_dataset_key == key and self._last_dataset_df is not None:
            return self._last_dataset_df

        df = self.fs_provider.load_dataframe(name, start_time=start_time, end_time=end_time, device=device)
        self._last_dataset_key = key
        self._last_dataset_df = df
        return df

    def list_devices(self):
        return list(self.collectors.keys())
