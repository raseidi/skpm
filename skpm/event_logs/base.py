import os
from typing import Any, Union
from urllib.error import URLError
from pandas import DataFrame
import pandas as pd

from skpm.event_logs.utils import download_and_extract_archive


class EventLog:
    """
    Base class for event logs.
    It provides the basic structure for downloading, preprocessing, and splitting.
    Furthermore, it provides the basic structure for caching the logs.

    Args:
        root_path (str, optional): Path where the event log will be stored.
            Defaults to "./data".
        config (Union[str, dict], optional): Configuration of the event log.
            Defaults to "default" (it just renames a few columns in the current version).
        transforms (Any, optional): Transformations to be applied to the event log.
            Defaults to None. To be implemented.
        kwargs: Additional arguments to be passed to the base class.
    """

    def __init__(
        self,
        root_path: str = "./data",
        config: Union[str, dict] = "default",  # ToDo
        transforms: Any = None,  # ToDo
        **kwargs,
    ) -> None:
        self.root_path = root_path
        if config == "default":
            self.config = {
                "case:concept:name": "case_id",
                "concept:name": "activity",
                "time:timestamp": "timestamp",
                "org:resource": "resource",
            }
        else:
            self.config = config

    def __repr__(self) -> str:
        head = "Event Log " + self.__class__.__name__
        body = [f"Number of events: {self.__len__()}"]
        if self.root_path is not None:
            body.append(
                f"Event log location: {os.path.join(self.root_path, self.__class__.__name__)}"
            )
        body += "".splitlines()
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)

    def download(self, url: str) -> None:
        if self._check_raw():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        try:
            print(f"Downloading {self.__class__.__name__}")
            download_and_extract_archive(
                url=url, root=self.raw_folder, filename=self.log_name
            )
        except URLError as error:
            print(f"Failed to download (trying next):\n{error}")
        finally:
            print()
            
    def _base_preprocess(self, log: DataFrame):
        log = log.rename(columns=self.config)
        return log


    def _check_raw(self):
        return os.path.exists(self.raw_log)

    @property
    def log_name(self):
        return self.__class__.__name__

    @property
    def raw_folder(self):
        return os.path.join(self.root_path, self.__class__.__name__, "raw")

    @property
    def raw_log(self):
        return os.path.join(self.raw_folder, "log.parquet")

    @property
    def cache_folder(self):
        return os.path.join(self.root_path, self.__class__.__name__, "cache")

    @property
    def train_file(self):
        return os.path.join(self.cache_folder, "split", "train.parquet")

    @property
    def test_file(self):
        return os.path.join(self.cache_folder, "split", "test.parquet")

class LogXES(EventLog):
    pass


class LogOCEL(EventLog):
    pass


class AsDataframeMixin:
    event_id_col: str = "event_id"
    case_id_col: str = "case_id"
    activity_col: str = "activity"
    timestamp_col: str = "timestamp"
    resource_col: str = "resource"
    
    cols_to_rename = {
        "case:concept:name": case_id_col,
        "concept:name": activity_col,
        "time:timestamp": timestamp_col,
        "org:resource": resource_col,
    }
    
    def __init_subclass__(cls) -> None:
        cls.base_preprocess()
        
    
    def base_preprocess(cls):
        cls.log = cls.log.rename(columns=cls.cols_to_rename)
        cls.log[cls.timestamp_col] = pd.to_datetime(cls.log[cls.timestamp_col], format='mixed')