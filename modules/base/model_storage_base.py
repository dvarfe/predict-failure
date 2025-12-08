from abc import ABC, abstractmethod

class AbstractModelStorage(ABC):
    @abstractmethod
    def save(self, model, name: str):
        pass

    @abstractmethod
    def load(self, name: str):
        pass

    @abstractmethod
    def list_models(self) -> list:
        pass