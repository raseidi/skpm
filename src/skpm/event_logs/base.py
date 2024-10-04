import os

import pandas as pd

from skpm.config import EventLogConfig as elc
from skpm.event_logs.parser import read_xes
from .download import download_url
from .extract import extract_gz


class BasePreprocessing:
    def preprocess(self):
        """
        Preprocess the event log by converting the timestamp column to
        datetime format.
        """
        self.log[elc.timestamp] = pd.to_datetime(
            self.log[elc.timestamp], utc=True, format="mixed"
        )


class TUEventLog(BasePreprocessing):
    """
    Base class for event logs from the 4TU repository.

    It provides the basic structure for downloading, preprocessing, and
    splitting
    Furthermore, it provides the basic structure for caching the logs.

    Event logs from the 4tu repository [1] are downloaded as .xes.gz files
    and then converted to parquet files. The parquet files are then used to
    load the event logs.
    By default, we keep the .xes files in the raw folder

    Args:
    -----
        root_path (str, optional): Path where the event log will be stored.
            Defaults to "./data".
        config (Union[str, dict], optional): Configuration of the event log.
            Defaults to "default" (it just renames a few columns in the
            current version).
        transforms (Any, optional): Transformations to be applied to the event
        log.
            Defaults to None. To be implemented.
        kwargs: Additional arguments to be passed to the base class.

    References:
    -----------
    [1] 4TU Research Data: https://data.4tu.nl/
    """

    url: str = None
    md5: str = None
    file_name: str = None
    meta_data: str = None  # TODO: download DATA.xml from the 4TU repository

    _unbiased_split_params: dict = None

    def __init__(
        self,
        root_folder: str = "./data",
        save_as_pandas: bool = True,
        train_set: bool = True,
        file_path: str = None,
    ) -> None:
        """
        Initialize the TUEventLog object.

        TODO: file_name, file_path, root_folder ??

        Parameters
        ----------
        root_folder : str, optional
            Path where the event log will be stored. Defaults to "./data".
        save_as_pandas : bool, optional
            Whether to save the event log as a Pandas DataFrame. Defaults to
            True.
        train_set : bool, optional
            Whether the event log is for the training set. Defaults to True.
        file_path : str, optional
            Path to the event log file. If None, the file will be downloaded.
            Defaults to None.
        """
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
        """
        str: Path to the event log file.
        """
        return self._file_path

    @file_path.setter
    def file_path(self, value):
        self._file_path = value

    @property
    def unbiased_split_params(self) -> dict:
        """
        dict: Parameters for the unbiased split of the event log.
        """
        if self._unbiased_split_params is None:
            raise ValueError("Unbiased split parameters not supported.")
        return self._unbiased_split_params

    def __len__(self):
        """
        Get the number of events in the event log.

        Returns
        -------
        int
            Number of events in the event log.
        """
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
        """
        Read the event log from the file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the event log data.
        """
        if self.file_path.endswith(".xes"):
            log = read_xes(self.file_path, as_df=True)

            if self.save_as_pandas:
                new_file_path = self.file_path.replace(
                    ".xes", elc.default_file_format
                )
                if elc.default_file_format == ".parquet":
                    log.to_parquet(new_file_path)
                else:
                    raise ValueError("File format not implemented.")
                os.remove(self.file_path)
                self.file_path = new_file_path

        elif self.file_path.endswith(elc.default_file_format):
            log = pd.read_parquet(self.file_path)
        else:
            raise ValueError("File format not implemented.")

        return log

    def __repr__(self) -> str:
        """
        Return a string representation of the TUEventLog object.

        Returns
        -------
        str
            String representation of the TUEventLog object.
        """
        head = "Event Log " + self.__class__.__name__
        body = [f"Number of events: {self.__len__()}"]
        if self.file_path is not None:
            body.append(
                f"Event log location: {os.path.normpath(self.file_path)}"
            )
        body += "".splitlines()
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)


class TUOCEL:
    pass
