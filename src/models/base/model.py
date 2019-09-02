from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def save(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError()
