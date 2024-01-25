import re
import os
import typing as t
from urllib.error import URLError
from warnings import warn
import pandas as pd

from skpm.event_logs.extract import extract_gz
from skpm.event_logs.download import download_url
from skpm.config import EventLogConfig as elc


class BasePreprocessing:
    def preprocess(self):
        self.log[elc.timestamp] = pd.to_datetime(
            self.log[elc.timestamp], format="mixed"
        )


class TUEventLog(BasePreprocessing):
    """
    Base class for event logs from the 4TU repository.

    It provides the basic structure for downloading, preprocessing, and splitting
    Furthermore, it provides the basic structure for caching the logs.

    Event logs from the 4tu repository are downloaded as .xes.gz files
    and then converted to parquet files. The parquet files are then used to
    load the event logs.
    By default, we keep the .xes files in the raw folder

    Args:
        root_path (str, optional): Path where the event log will be stored.
            Defaults to "./data".
        config (Union[str, dict], optional): Configuration of the event log.
            Defaults to "default" (it just renames a few columns in the current version).
        transforms (Any, optional): Transformations to be applied to the event log.
            Defaults to None. To be implemented.
        kwargs: Additional arguments to be passed to the base class.
    """

    url: str = None
    md5: str = None
    file_name: str = None
    meta_data: str = None  # TODO: download DATA.xml from the 4TU repository

    def __init__(
        self,
        root_folder: str = "./data",
        save_as_pandas: bool = True,
        train_set: bool = True,
        file_path: str = None,
    ) -> None:
        super().__init__()
        self.root_folder = root_folder
        self.save_as_pandas = save_as_pandas
        self.train_set = train_set

        if file_path is None:
            self._file_path = os.path.join(
                self.root_folder,
                self.__class__.__name__,
                self.file_name.replace(".gz", "").replace(
                    ".xes", elc.default_file_format
                ),
            )
        else:
            self._file_path = file_path

        if not os.path.exists(self.file_path):
            self.download()

        self.log = self.read_log()
        self.preprocess()

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value):
        self._file_path = value

    def __len__(self):
        return len(self.log)

    def download(self) -> None:
        """Generic method to download the event log from the 4TU Repository.

        It downloads the event log from the url, uncompresses
        it, and stores it. It can be overwritten by the
        subclasses if needed.
        """
        destination_folder = os.path.join("data", self.__class__.__name__)
        print(f"Downloading {destination_folder}")
        path = download_url(
            url=self.url, folder=destination_folder, file_name=self.file_name
        )
        if path.endswith(".xes"):
            self.file_path = path
            return

        if path.endswith(".gz"):
            self.file_path = extract_gz(
                path=path, folder=os.path.dirname(destination_folder)
            )
        # TODO: elif other formats
        os.remove(path)

    def read_log(self) -> pd.DataFrame:
        if self.file_path.endswith(".xes"):
            import pm4py

            log = pm4py.read_xes(self.file_path)

            if self.save_as_pandas:
                new_file_path = self.file_path.replace(".xes", elc.default_file_format)
                log.to_parquet(new_file_path)
                os.remove(self.file_path)
                self.file_path = new_file_path

        elif self.file_path.endswith(elc.default_file_format):
            log = pd.read_parquet(self.file_path)  # , engine="fastparquet")
            # which engine is better?
            # log = pd.read_parquet(self.file_path, engine="fastparquet")

        # log = self._base_preprocess(log)
        return log

        # Ideally we want to standardize the train/test sets
        # see https://github.com/hansweytjens/predictive-process-monitoring-benchmarks
        # train, test = self.split_log(log)
        # train.to_parquet(self.train_file, index=False)
        # test.to_parquet(self.test_file, index=False)

        # return train if self.train else test

    def __repr__(self) -> str:
        head = "Event Log " + self.__class__.__name__
        body = [f"Number of events: {self.__len__()}"]
        if self.file_path is not None:
            body.append(f"Event log location: {os.path.normpath(self.file_path)}")
        body += "".splitlines()
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)


class TUOCEL:
    pass
