from abc import ABC, abstractmethod
import pandas as pd


class ExtractProcess(ABC):
    """
    base abstract class to define different variants of feature extractions
    different sources of data need to be parsed differently
    """
    @abstractmethod
    def extract_features(self) -> pd.Series:
        """
        this method gets called on every variant and is responsible to
        parse that specific variant of data to extract features
        :return:
        """
        raise NotImplemented()
