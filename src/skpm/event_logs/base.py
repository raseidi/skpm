import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from skpm.event_logs.parser import read_xes
from urllib import request


class EventLogConfig:
    """
    Global configuration manager for event log column mappings and file formats.

    This class supports global configuration following the classic XES naming.
    Changes to the global config  automatically propagate to all instances.
    SkPM will never rename columns, so users should ensure their data matches the
    expected column names.

    Examples
    --------
    # Set global configuration
    EventLogConfig.set_global_config(case_id='case_id', activity='activity')

    # Reset global configuration to defaults
    EventLogConfig.reset_global_config()
    """

    # Global default configuration - shared across all instances
    _GLOBAL_CONFIG = {
        "case_id": "case:concept:name",
        "activity": "concept:name",
        "timestamp": "time:timestamp",
    }

    @property
    def case_id(self) -> str:
        """Case ID column name."""
        return self._GLOBAL_CONFIG["case_id"]

    @property
    def activity(self) -> str:
        """Activity column name."""
        return self._GLOBAL_CONFIG["activity"]

    @property
    def timestamp(self) -> str:
        """Timestamp column name."""
        return self._GLOBAL_CONFIG["timestamp"]

    def to_dict(self) -> Dict[str, str]:
        """Convert current configuration to dictionary."""
        return {
            "case_id": self.case_id,
            "activity": self.activity,
            "timestamp": self.timestamp,
            "resource": self.resource,
            "default_file_format": self.default_file_format,
        }

    @classmethod
    def get_global_config(cls) -> Dict[str, str]:
        """Get the current global configuration."""
        return cls._GLOBAL_CONFIG.copy()

    @classmethod
    def set_global_config(
        cls,
        case_id: Optional[str] = None,
        activity: Optional[str] = None,
        timestamp: Optional[str] = None,
        resource: Optional[str] = None,
    ) -> None:
        """
        Set the global configuration.

        This will affect all instances that don't have explicit overrides.

        Parameters
        ----------
        case_id : Optional[str]
            Case ID column name.
        activity : Optional[str]
            Activity column name.
        timestamp : Optional[str]
            Timestamp column name.
        resource : Optional[str]
            Resource column name.
        default_file_format : Optional[str]
            Default file format.
        """
        if case_id is not None:
            cls._GLOBAL_CONFIG["case_id"] = case_id
        if activity is not None:
            cls._GLOBAL_CONFIG["activity"] = activity
        if timestamp is not None:
            cls._GLOBAL_CONFIG["timestamp"] = timestamp
        if resource is not None:
            cls._GLOBAL_CONFIG["resource"] = resource

    @classmethod
    def reset_global_config(cls) -> None:
        """Reset global configuration to defaults."""
        cls._GLOBAL_CONFIG = {
            "case_id": "case:concept:name",
            "activity": "concept:name",
            "timestamp": "time:timestamp",
        }

    def __repr__(self) -> str:
        """String representation showing current configuration."""
        config = self.to_dict()

        lines = ["EventLogConfig:"]
        for key, value in config.items():
            lines.append(f"  {key}: '{value}'")

        return "\n".join(lines)


class EventLog:
    """Base class for event log handling with common functionality."""

    config: EventLogConfig = EventLogConfig()
    _dataframe: Optional[pd.DataFrame] = None

    def __init__(self, dataframe: Optional[pd.DataFrame] = None):
        self._dataframe = dataframe

        if self._dataframe is not None:
            self.preprocess()

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @property
    def case_id(self) -> str:
        return self.config.case_id

    @property
    def activity(self) -> str:
        return self.config.activity

    @property
    def timestamp(self) -> str:
        return self.config.timestamp

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
        Folder to cache downloaded files (default is Path.home() / "skpm" / "event_logs")
    Attributes
    ----------
    url : Optional[str]
        URL to download the event log from (should be set in subclasses)
    file_name : Optional[str]
        Name of the file to download (should be set in subclasses)
    file_path : Path
        Path to the event log file (will be set based on file_name and cache_folder)
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
        """
        Initialize TUEventLog.

        Parameters
        ----------
        cache_folder : Union[str, Path]
            Folder to cache downloaded files (should be provided if file_path is None)
        default_file_format : str
            Default file format for the event log for caching (options: 'parquet', 'csv')
        """
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
            self.file_path = self.cache_folder / self.file_name  # xes file
        else:
            self.file_path = cached_file

        if not self.file_path.exists():
            self._download()

        df = self._read_log()
        if df is None or df.empty:
            raise ValueError(f"Failed to load data from {self._file_path}")

        super().__init__(dataframe=df)

    def _download(self) -> None:
        if not self.url or not self.file_name:
            raise ValueError("URL and file_name must be set for download")

        with request.urlopen(request.Request(self.url)) as response:
            with open(self.file_path, "wb") as fh:
                while True:
                    chunk = response.read(1024 * 32)
                    if not chunk:
                        break
                    fh.write(chunk)

        if self.file_path.suffix == ".gz":
            self._extract_gz()

    def _read_log(self) -> pd.DataFrame:
        """Read event log from file with format detection."""
        file_path = self.file_path

        if file_path.suffix == ".xes":
            log = read_xes(str(file_path))

            # delete xes and save as pandas
            self.file_path.unlink()
            self.file_path = self.cache_folder / self.cached_file_name

            if self.default_file_format == ".parquet":
                log.to_parquet(self.file_path, index=False)
            elif self.default_file_format == ".csv":
                log.to_csv(self.file_path, index=False)
            else:
                raise ValueError(
                    f"Unsupported format: {self.default_file_format}"
                )

            return log

        elif file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _extract_gz(self):
        r"""Extracts a gz archive to a specific folder.

        Args:
            path (str): The path to the tar archive.
            folder (str): The folder.
            log (bool, optional): If :obj:`False`, will not print anything to the
                console. (default: :obj:`True`)
        """
        import gzip

        # self.file_path = a/b/c.xez.gz
        with gzip.open(self.file_path, "r") as r:
            with open(self.file_path.with_suffix(""), "wb") as w:
                w.write(r.read())

        # self.file_path = a/b/c.xez
        old_file_path = self.file_path
        self.file_path = self.file_path.with_suffix("")
        old_file_path.unlink()

    @property
    def unbiased_split_params(self) -> Dict[str, Any]:
        """Parameters for unbiased data splitting."""
        if self._unbiased_split_params is None:
            raise ValueError(
                f"Unbiased split not available for {self.__class__.__name__}"
            )
        return self._unbiased_split_params
