from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Model(ABC):
    @abstractmethod
    def save(self):
        raise NotImplementedError()

    @abstractmethod
    def train(
            self,
            x: pd.DataFrame,
            y: pd.Series,
            batch_size: int,
            epochs: int,
            verbose: bool,
    ) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def predict(
            self,
            x: pd.DataFrame,
            batch_size: int,
            verbose: int,
    ):
        raise NotImplementedError()

    @abstractmethod
    def describe(self):
        raise NotImplementedError()
