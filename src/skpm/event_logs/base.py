import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib import request

import pandas as pd

from skpm.base import EventLogConfigMixin
from skpm.event_logs.parser import read_xes

class EventLog(EventLogConfigMixin):
    """Base class for event log handling with common functionality.
    
    Every class that handles event logs should inherit from this class.
    This class provides methods for preprocessing, validation, and summary statistics
    of event logs. It also defines the expected structure of an event log,
    including case ID, activity, and timestamp columns.
    """
    _dataframe: Optional[pd.DataFrame] = None

    def __init__(self, dataframe: Optional[pd.DataFrame] = None):
        self._dataframe = dataframe

        if self._dataframe is not None:
            self.preprocess()

    @property
    def dataframe(self) -> pd.DataFrame:
        """Get the event log DataFrame."""
        return self._dataframe

    def preprocess(self) -> None:
        """Preprocess the dataframe by converting timestamps."""
        if self.dataframe is None:
            raise ValueError("No dataframe loaded to preprocess")

        # More efficient datetime conversion with error handling
        self._dataframe[self.timestamp] = pd.to_datetime(
            self._dataframe[self.timestamp],
            utc=True,
            format="mixed",
            errors="coerce",
        )

        # Log any failed conversions
        null_count = self.dataframe[self.timestamp].isnull().sum()
        if null_count > 0:
            logging.warning(f"Failed to convert {null_count} timestamps")

        self._dataframe = self._dataframe.sort_values(
            by=[self.timestamp], ascending=True
        ).reset_index(drop=True)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the event log."""
        df = self.dataframe

        return {
            "n_cases": df[self.case_id].nunique(),
            "n_events": len(df),
            "n_activities": (
                df[self.activity].nunique()
                if self.activity in df.columns
                else 0
            ),
        }

    def __repr__(self) -> str:
        """String representation of the BaseEventLog object."""
        if self.dataframe is None:
            return f"{self.__class__.__name__} (not loaded)"

        stats = self.get_summary_stats()

        lines = [
            f"{self.__class__.__name__} Event Log",
            f"    Cases: {stats['n_cases']:,}",
            f"    Events: {stats['n_events']:,}",
            f"    Activities: {stats['n_activities']:,}",
        ]

        return "\n".join(lines)


class TUEventLog(EventLog):
    """
    Class for downloading and managing 4TU repository data. This class
    should be subclassed for specific event logs. The data will be either
    downloaded from the 4TU repository or read from a default cached folder.

    Parameters
    ----------
    cache_folder : Union[str, Path], optional
        Folder to cache downloaded files (default is Path.home() / "skpm" / "event_logs")4
    default_file_format : str, optional
        Default file format for the event log (options: "parquet", "csv"; default is "parquet")
    Attributes
    ----------
    url : Optional[str]
        Remote URL to download the event log from (should be set in subclasses)
    cached_file_name : Path
        Name of the cached file (will be set based on class name and default file format)
    cache_folder : Path
        Folder where the event log file will be cached (default is Path.home() / "skpm" / "event_logs" / class name)
    file_name : Optional[str]
        Name of the file to download (should be set in subclasses)
    file_path : Path
        Path to the event log file (will be set based on file_name and cache_folder)
    _dataframe : Optional[pd.DataFrame]
        DataFrame containing the event log data (will be set after loading)
    unbiased_split_params : Optional[Dict[str, Any]]
        Parameters for unbiased data splitting (should be set in subclasses)
    """

    url: Optional[str] = None
    file_name: Optional[str] = None
    _unbiased_split_params: Optional[Dict[str, Any]] = None

    file_formats = {
        "parquet": ".parquet",
        "csv": ".csv",
    }

    def __init__(
        self,
        cache_folder: Union[str, Path] = None,
        default_file_format: str = "parquet",
    ) -> None:
        self.default_file_format = self.file_formats.get(
            default_file_format, ".parquet"
        )

        if cache_folder is not None:
            cache_folder = Path(cache_folder)
            if not cache_folder.is_dir():
                raise ValueError(
                    f"cache_folder must be a directory: {cache_folder}"
                )
        self.cache_folder = (
            Path(cache_folder) / Path(self.__class__.__name__)
            if cache_folder
            else Path.home()
            / "skpm"
            / "event_logs"
            / Path(self.__class__.__name__)
        )

        self._ensure_data_loaded()

    def _ensure_data_loaded(self) -> None:
        """Ensure data is downloaded and loaded."""
        if self._dataframe is not None:
            return

        self.cached_file_name = Path(self.__class__.__name__).with_suffix(
            self.default_file_format
        )
        cached_file = self.cache_folder / self.cached_file_name
        if not cached_file.exists():
            self.cache_folder.mkdir(exist_ok=True)
            self.file_path = []
            for file in self.file_name:
                self.file_path.append(self.cache_folder / file)  # xes file
        else:
            self.file_path = [cached_file]

        if not self.file_path[0].exists():
            self._download()

        df = self._read_log()
        if df is None or df.empty:
            raise ValueError(f"Failed to load data from {self.file_path}")

        super().__init__(dataframe=df)

    def _download(self) -> None:
        if not self.url or not self.file_name:
            raise ValueError("URL and file_name must be set for download")

        for ix, url in enumerate(self.url):
            with request.urlopen(request.Request(url)) as response:
                with open(self.file_path[ix], "wb") as fh:
                    while True:
                        chunk = response.read(1024 * 32)
                        if not chunk:
                            break
                        fh.write(chunk)

        if self.file_path[0].suffix == ".gz":
            self._extract_gz()

    def _read_log(self) -> pd.DataFrame:
        """Read event log from file with format detection."""
        df = pd.DataFrame()
        for ix, file_path in enumerate(self.file_path):
            if file_path.suffix == ".xes":
                log = read_xes(str(file_path))
                log['log_version'] = file_path.stem  # Add log version from file name
                df = pd.concat([df, log], ignore_index=True)

                # delete xes and save as pandas
                file_path.unlink()

                if ix == len(self.file_path) - 1:
                    file_path = self.cache_folder / self.cached_file_name
                    if self.default_file_format == ".parquet":
                        df.to_parquet(file_path, index=False)
                    elif self.default_file_format == ".csv":
                        df.to_csv(file_path, index=False)
                    else:
                        raise ValueError(
                            f"Unsupported format: {self.default_file_format}"
                        )

            elif file_path.suffix == ".parquet":
                log = pd.read_parquet(file_path)
                df = pd.concat([df, log], ignore_index=True)
            elif file_path.suffix == ".csv":
                log = pd.read_csv(file_path)
                df = pd.concat([df, log], ignore_index=True)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        return df

    def _extract_gz(self):
        """Extracts a gz archive to a specific folder."""
        import gzip

        for file_path in self.file_path:
            # self.file_path = a/b/c.xez.gz
            with gzip.open(file_path, "r") as r:
                with open(file_path.with_suffix(""), "wb") as w:
                    w.write(r.read())

            # self.file_path = a/b/c.xez
            old_file_path = file_path
            self.file_path = (file_path.with_suffix(""),)
            old_file_path.unlink()

    @property
    def unbiased_split_params(self) -> Dict[str, Any]:
        """Parameters for unbiased data splitting."""
        if self._unbiased_split_params is None:
            raise ValueError(
                f"Unbiased split not available for {self.__class__.__name__}"
            )
        return self._unbiased_split_params
