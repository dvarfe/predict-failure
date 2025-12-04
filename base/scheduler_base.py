from abc import ABC, abstractmethod
from typing import Any


class SchedulerBase(ABC):
    """Базовый абстрактный класс для планировщика задач
    """

    def __init__(self, name: str = ""):
        self.name = name

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        raise NotImplementedError()

    @abstractmethod
    def add_job(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def remove_job(self, job_id: str) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def get_job(self, job_id: str) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def get_jobs(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def apply_schedule(self, schedule_config: dict, job_func: Any) -> None:
        raise NotImplementedError()
