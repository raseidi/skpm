import os
from urllib.error import URLError
import pandas as pd

from skpm.event_logs.utils import download_and_extract_archive


class EventLog:
    def __init__(
        self,
        root_path="./data",
        config="default",  # ToDo
        transforms=None,  # ToDo
        **kwargs,
    ) -> None:
        self.root_path = root_path
        if config == "default":
            self.config = {
                "case_col": "case:concept:name",
                "activity_col": "concept:name",
                "timestamp_col": "time:timestamp",
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

    def download(self, url) -> None:
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

    def _check_raw(self):
        return os.path.exists(self.raw_log)

    def _base_preprocess(self, log):
        log = log.rename(columns=self.config)
        return log

    @property
    def log_name(self):
        return self.__class__.__name__

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root_path, self.__class__.__name__, "raw")

    @property
    def raw_log(self):
        return os.path.join(self.raw_folder, "log.parquet")

    @property
    def cache_folder(self) -> str:
        return os.path.join(self.root_path, self.__class__.__name__, "cache")

    @property
    def train_file(self):
        return os.path.join(self.cache_folder, "split", "train.parquet")

    @property
    def test_file(self):
        return os.path.join(self.cache_folder, "split", "test.parquet")
