from typing import Type, List, Any, Iterator
from src.features.base.extract_process.extract_process import ExtractProcess


class DataSource:
    """
    adapter class to bundle any variant of data processor with that variant
    specific arguments. This essentially provides the flexibility to define
    several feature ExtractProcess variants with differing arguments and
    compiling them together in a way that allows extracting with them with
    only shared arguments
    """
    Processor: Type[ExtractProcess]
    processor_args: dict
    epoch_arg_name: str

    def __init__(
            self,
            processor: Type[ExtractProcess],
            processor_args: dict,
            epoch_arg_name: str
    ):
        """
        constructor for DataSource
        :param processor: any class that extends ExtractProcess
        :param processor_args: any additional arguments passed on as dict
        with argument_names as keys
        :param epoch_arg_name: in what name should the actual data from which
        the data needs to be extracted be passed on as?
        """
        self.Processor = processor
        self.processor_args = processor_args
        self.epoch_arg_name = epoch_arg_name

    def __get_instance(self, epoch: Any) -> ExtractProcess:
        """
        builds an instance of class that extends ExtractProcess
        :param epoch: the actual data from which the features need to be
        extracted from
        :return: an instance of class that extends ExtractProcess
        """
        args = dict(self.processor_args)
        args[self.epoch_arg_name] = epoch
        # noinspection PyArgumentList
        return self.Processor(**args)

    def get_processors(self, epochs: List[Any]) -> Iterator[ExtractProcess]:
        """
        maps every epoch to an extractor
        :param epochs: list of data epochs
        :return: list of instances of class extending ExtractProcess bundled
        with data that it'll process
        """
        return map(self.__get_instance, epochs)
