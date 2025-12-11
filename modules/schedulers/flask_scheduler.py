from flask_apscheduler import APScheduler
from typing import Any
from datetime import datetime, timedelta
from ..base.scheduler_base import SchedulerBase


class FlaskScheduler(SchedulerBase):
    def __init__(self, app, name, enabled, interval_value, interval_unit, selected_collectors):
        self.aps = APScheduler()
        self._app = app
        self.aps.init_app(app)
        self.name = name
        self.enabled = enabled
        self.interval_value = interval_value
        self.interval_unit = interval_unit
        self.selected_collectors = selected_collectors

    def start(self):
        self.aps.start()

    def shutdown(self, wait=True):
        self.aps.shutdown(wait=wait)

    def add_job(self, *args, **kwargs):
        return self.aps.add_job(*args, **kwargs)

    def remove_job(self, job_id):
        return self.aps.remove_job(job_id)

    def get_job(self, job_id):
        return self.aps.get_job(job_id)

    def get_jobs(self):
        return self.aps.get_jobs()

    def apply_schedule(self, schedule_config: dict, job_func: Any) -> None:
        """Применить планировщик задач на основе schedule_config.

        `job_func` вызывается с `selected_collectors` в качестве kwargs.
        """
        # Для начала удалить существующие задачи, если они есть
        if self.get_job('data_collection'):
            self.remove_job('data_collection')

        if schedule_config.get('enabled', False) and schedule_config.get('selected_collectors', []):
            interval_value = schedule_config.get('interval_value', 5)
            interval_unit = schedule_config.get('interval_unit', 'minutes')

            scheduler_kwargs = {}
            if interval_unit == 'seconds':
                scheduler_kwargs['seconds'] = interval_value
            elif interval_unit == 'minutes':
                scheduler_kwargs['minutes'] = interval_value
            elif interval_unit == 'hours':
                scheduler_kwargs['hours'] = interval_value
            elif interval_unit == 'days':
                scheduler_kwargs['days'] = interval_value

            ###########################################################################
            # Ограничить число запусков
            N_TIMES = 5
            if interval_unit == 'seconds':
                end_time = datetime.now() + timedelta(seconds=N_TIMES * interval_value)
            elif interval_unit == 'minutes':
                end_time = datetime.now() + timedelta(minutes=N_TIMES * interval_value)
            elif interval_unit == 'hours':
                end_time = datetime.now() + timedelta(hours=N_TIMES * interval_value)
            elif interval_unit == 'days':
                end_time = datetime.now() + timedelta(days=N_TIMES * interval_value)
            ############################################################################

            self.add_job(
                id='data_collection',
                func=job_func,
                trigger='interval',
                kwargs={'selected_collectors': schedule_config.get('selected_collectors', [])},
                replace_existing=True,
                end_date=end_time,
                **scheduler_kwargs,
            )
