import pandas as pd
from core.config import ConfigManager
from core.providers import DICT_DATA_PROVIDERS, DEFAULT_PROVIDER_PARAMS, DEFAULT_PROVIDER_TYPE
from schedulers import DICT_SCHEDULERS, DEFAULT_SCHEDULER_PARAMS
from collectors import DICT_COLLECTORS


class SystemManager:
    def __init__(self, app=None):
        self.config_manager = ConfigManager()
        self.data = None
        self.setup_config(app)

    def setup_collectors(self):
        self.collectors = {}
        DICT_COLLECT_FOR_OS = DICT_COLLECTORS.get(self.config_manager.get_system())
        global_cfg = self.config_manager.get_config().get(
            'default_data_provider', DEFAULT_PROVIDER_PARAMS)
        global_type = list(global_cfg.keys())[0]
        for name, config in self.config_manager.get_collectors().items():
            collector_obj = DICT_COLLECT_FOR_OS.get(name)(config)

            p_cfg = config.get('data_provider', global_cfg)
            p_type = list(p_cfg.keys())[0] if p_cfg else global_type
            p_cfg = {**p_cfg[p_type], **{'device_name': name}}
            provider = self.setup_provider(p_type, p_cfg)
            collector_obj.set_data_provider(provider)

            self.collectors[name] = collector_obj

    def setup_provider(self, provider_type: str, p_cfg: dict):
        Provider = DICT_DATA_PROVIDERS.get(provider_type)
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

    def register_model(self, name: str, model_cls):
        self.models[name] = model_cls

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
