import platform
import json
import os

from collectors import DICT_COLLECTORS
from models import DICT_MODELS
from schedulers import DEFAULT_SCHEDULER_PARAMS
from core.providers import DEFAULT_PROVIDER_TYPE, DEFAULT_PROVIDER_PARAMS

def save_config(config, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


def load_config(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    system = platform.system()
    enabled_collectors = list(DICT_COLLECTORS.get(system, {}).keys())
    config = {"system": system, "enabled_collectors": enabled_collectors}
    config["collectors"] = {c: {} for c in enabled_collectors}
    config["models"] = list(DICT_MODELS.keys())
    config["scheduler"] = DEFAULT_SCHEDULER_PARAMS
    config["default_data_provider"] = DEFAULT_PROVIDER_PARAMS.get(DEFAULT_PROVIDER_TYPE, {})
    save_config(config, path)
    return config


class ConfigManager:
    def __init__(self, config_path="storage/configs/config.json"):
        self.path = config_path
        self.config = load_config(self.path)

    def get_system(self):
        return self.config.get("system", "")

    def get_config(self):
        return self.config

    def get_collector_config(self, name):
        configs = self.get_collectors()
        return configs.get(name, {})

    def get_collectors(self):
        return self.config.get("collectors", {})

    def update_collector_config(self, name, new_config):
        if "collectors" not in self.config:
            self.config["collectors"] = {}
        self.config["collectors"][name] = new_config
        self.save_config(self.config, self.path)

    def get_scheduler(self):
        return self.config.get("scheduler", {})

    def update_scheduler_config(self, new_config):
        self.config["scheduler"] = new_config
        self.save_config(self.config, self.path)

    def save_config(self, config: dict = None, path: str = "storage/configs/config.json"):
        save_config(config, path)

    def __del__(self):
        self.save_config(self.config, self.path)
