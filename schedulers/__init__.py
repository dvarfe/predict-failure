from typing import Dict, Type

from core.scheduler import FlaskScheduler

DICT_SCHEDULERS: Dict[str, Type] = {
    'flask': FlaskScheduler,
}

DEFAULT_SCHEDULER_PARAMS = {"name": "flask",
                            "enabled": False,
                            "interval_value": 5,
                            "interval_unit": "minutes",
                            "selected_collectors": []}
