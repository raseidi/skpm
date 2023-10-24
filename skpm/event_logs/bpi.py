import os
from typing import Any
from pandas import read_parquet
from .base import EventLog


class BPI12(EventLog):
    """BPI Challenge 2012

    DOI: 10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f

    This is an event log of a loan application process from a Dutch financial
    institute. The process is concerned with handling loan applications for
    private individuals. The data is enriched with decision and execution
    information added by a process mining tool.

    Args:
        root_path (str, optional): Path where the event log will be stored.
            Defaults to "./data".
        train (bool, optional): If True, returns the training set. Otherwise,
            returns the test set. Defaults to True.
        kwargs: Additional arguments to be passed to the base class.
    """

    url = "https://data.4tu.nl/file/533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893"

    def __init__(
        self, root_path: str = "./data", train: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(root_path=root_path, **kwargs)
        self.train = train

        if self._check_cache():
            self.log = self._load_cache_log()
            return

        if not self._check_raw():
            self.download(self.url)

        self.log = self._load_log()

    def __len__(self):
        return len(self.log)

    def _check_cache(self):
        return os.path.exists(self.train_file)

    def _load_cache_log(self):
        file_path = self.train_file if self.train else self.test_file
        log = read_parquet(file_path)
        log = self._base_preprocess(log)
        return log

    def _load_log(self):
        log = read_parquet(self.log_file)
        log = self._base_preprocess(log)
        train, test = self.split_log(log)
        train.to_parquet(self.train_file, index=False)
        test.to_parquet(self.test_file, index=False)

        return train if self.train else test


class BPI17OCEL(EventLog):
    """BPI Challenge 2017 OCEL

    DOI: 10.4121/6889ca3f-97cf-459a-b630-3b0b0d8664b5.v1

    This dataset contains the result of transforming the BPI Challenge 2017
    event log from Event Graph format to Object-Centric Event Log (OCEL) format.
    The transformation is defined in the "Transforming Event Knowledge Graph
    to Object-Centric Event Logs: A Comparative Study for Multi-dimensional
    Process Analysis" paper.

    Args:
        root_path (str, optional): Path where the event log will be stored.
            Defaults to "./data".
        train (bool, optional): If True, returns the training set. Otherwise,
            returns the test set. Defaults to True.
        kwargs: Additional arguments to be passed to the base class.
    """

    url = "https://data.4tu.nl/file/6889ca3f-97cf-459a-b630-3b0b0d8664b5/5d5b9f89-7fa6-4c92-b6ac-04f854bdf92e"

    def __init__(
        self, root_path: str = "./data", train: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(root_path=root_path, **kwargs)
        self.train = train

        if not self._check_raw():
            self.download(self.url)

    def _load_log(self):
        raise NotImplementedError

    @property
    def raw_log(self):
        return os.path.join(self.raw_folder, f"{self.__class__.__name__}.jsonocel")
