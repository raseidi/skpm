from functools import cached_property
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from skpm.event_logs.parser import read_xes
from skpm.event_logs.download import download_url
from skpm.event_logs.extract import extract_gz

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
    
    # Update global config (affects all instances without overrides)
    EventLogConfig.update_global_config(timestamp='my_timestamp')
    
    # Reset global configuration to defaults
    EventLogConfig.reset_global_config()
    """
    
    # Global default configuration - shared across all instances
    _GLOBAL_CONFIG = {
        'case_id': "case:concept:name",
        'activity': "concept:name", 
        'timestamp': "time:timestamp",
        'resource': "org:resource",
        'default_file_format': ".parquet"
    }
        
    @property
    def case_id(self) -> str:
        """Case ID column name."""
        return self._GLOBAL_CONFIG['case_id']
    
    @property
    def activity(self) -> str:
        """Activity column name."""
        return self._GLOBAL_CONFIG['activity']
    
    @property
    def timestamp(self) -> str:
        """Timestamp column name."""
        return self._GLOBAL_CONFIG['timestamp']
    
    @property
    def resource(self) -> str:
        """Resource column name."""
        return self._GLOBAL_CONFIG['resource']
    
    @property
    def default_file_format(self) -> str:
        """Default file format."""
        return self._GLOBAL_CONFIG['default_file_format']

    
    def clear_override(self, *args) -> None:
        """
        Clear instance-specific overrides (revert to global config).
        
        Parameters
        ----------
        *args : str
            Parameter names to clear. If none provided, clears all overrides.
        """
        if not args:
            # Clear all overrides
            for key in self._overrides:
                self._overrides[key] = None
        else:
            # Clear specific overrides
            for key in args:
                if key in self._overrides:
                    self._overrides[key] = None
                else:
                    raise ValueError(f"Unknown configuration parameter: {key}")
    
    def to_dict(self) -> Dict[str, str]:
        """Convert current configuration to dictionary."""
        return {
            'case_id': self.case_id,
            'activity': self.activity,
            'timestamp': self.timestamp,
            'resource': self.resource,
            'default_file_format': self.default_file_format
        }
    
    @classmethod
    def get_global_config(cls) -> Dict[str, str]:
        """Get the current global configuration."""
        return cls._GLOBAL_CONFIG.copy()
    
    @classmethod
    def set_global_config(cls, 
                         case_id: Optional[str] = None,
                         activity: Optional[str] = None,
                         timestamp: Optional[str] = None,
                         resource: Optional[str] = None,
                         default_file_format: Optional[str] = None) -> None:
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
            cls._GLOBAL_CONFIG['case_id'] = case_id
        if activity is not None:
            cls._GLOBAL_CONFIG['activity'] = activity
        if timestamp is not None:
            cls._GLOBAL_CONFIG['timestamp'] = timestamp
        if resource is not None:
            cls._GLOBAL_CONFIG['resource'] = resource
        if default_file_format is not None:
            cls._GLOBAL_CONFIG['default_file_format'] = default_file_format
    
    @classmethod
    def update_global_config(cls, **kwargs) -> None:
        """
        Update global configuration with provided parameters.
        
        Parameters
        ----------
        **kwargs : dict
            Configuration parameters to update globally.
        """
        for key, value in kwargs.items():
            if key in cls._GLOBAL_CONFIG:
                cls._GLOBAL_CONFIG[key] = value
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    @classmethod
    def reset_global_config(cls) -> None:
        """Reset global configuration to defaults."""
        cls._GLOBAL_CONFIG = {
            'case_id': "case:concept:name",
            'activity': "concept:name",
            'timestamp': "time:timestamp", 
            'resource': "org:resource",
            'default_file_format': ".parquet"
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
        """DataFrame containing the event log data."""
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
    
    @property
    def resource(self) -> str:
        return self.config.resource
    
    def preprocess(self) -> None:
        """Preprocess the dataframe by converting timestamps."""
        if self.dataframe is None:
            raise ValueError("No dataframe loaded to preprocess")
        
        # More efficient datetime conversion with error handling
        self.dataframe[self.timestamp] = pd.to_datetime(
            self.dataframe[self.timestamp], 
            utc=True, 
            format="mixed",
            errors='coerce'
        )
        
        # Log any failed conversions
        null_count = self.dataframe[self.timestamp].isnull().sum()
        if null_count > 0:
            logging.warning(f"Failed to convert {null_count} timestamps")
                        
        self.dataframe = self.dataframe.sort_values(
            by=[self.timestamp],
            ascending=True
        ).reset_index(drop=True)
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the event log."""
        df = self.dataframe
        
        return {
            "n_cases": df[self.case_id].nunique(),
            "n_events": len(df),
            "n_activities": df[self.activity].nunique() if self.activity in df.columns else 0,
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
        
class FileManager:
    """Handles file operations for event logs."""
    
    def __init__(self, root_folder: Union[str, Path] = None):
        if root_folder is None:
            self.root_folder = Path.home() / "skpm" / "event_logs"
        else:
            self.root_folder = Path(root_folder)
            
    def get_file_path(self, class_name: str, file_name: str, 
                     file_format: str = ".parquet") -> Path:
        """Generate standardized file path."""
        processed_name = file_name.replace(".gz", "").replace(".xes", file_format)
        return self.root_folder / class_name / processed_name
    
    def ensure_directory(self, file_path: Path) -> None:
        """Ensure parent directory exists."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def download_and_extract(self, url: str, file_name: str, 
                           destination_folder: Path) -> Path:
        """Download and extract file, returning final path."""
        self.ensure_directory(destination_folder / file_name)
                
        downloaded_path = Path(download_url(
            url=url, 
            folder=str(destination_folder), 
            file_name=file_name
        ))
        
        if downloaded_path.suffix == ".xes":
            return downloaded_path
        
        if downloaded_path.suffix == ".gz":
            extracted_path = Path(extract_gz(
                path=str(downloaded_path), 
                folder=str(destination_folder.parent)
            ))
            downloaded_path.unlink()  # Remove compressed file
            return extracted_path
        
        return downloaded_path

class TUEventLog(EventLog):
    """
    Enhanced event log class for 4TU repository data.
    
    Features:
    - Lazy loading of data
    - Better error handling
    - Improved file management
    - Caching support
    - Performance optimizations
    """
    
    url: Optional[str] = None
    md5: Optional[str] = None
    file_name: Optional[str] = None
    meta_data: Optional[str] = None
    _unbiased_split_params: Optional[Dict[str, Any]] = None
    
    def __init__(
        self,
        root_folder: Union[str, Path] = None,
        save_as_pandas: bool = True,
        file_path: Optional[Union[str, Path]] = None,
        lazy_load: bool = False,
    ) -> None:
        """
        Initialize TUEventLog.
        
        Parameters
        ----------
        root_folder : Union[str, Path]
            Root directory for data storage
        save_as_pandas : bool
            Whether to convert XES to pandas format
        file_path : Optional[Union[str, Path]]
            Custom file path (skips download if provided)
        lazy_load : bool
            Whether to defer loading until accessed
        """
        self.save_as_pandas = save_as_pandas
        self.file_manager = FileManager(root_folder)
        
        # Set up file path
        if file_path:
            self._file_path = Path(file_path)
        else:
            if not self.file_name:
                raise ValueError("file_name must be set for auto-download")
            self._file_path = self.file_manager.get_file_path(
                self.__class__.__name__, 
                self.file_name,
                self.config.default_file_format
            )
        
        # Load immediately or defer based on lazy_load flag
        if not lazy_load:
            self._ensure_data_loaded()
    
    def _ensure_data_loaded(self) -> None:
        """Ensure data is downloaded and loaded."""
        if self._dataframe is not None:
            return
        
        if not self._file_path.exists():
            self._download()
        
        self._dataframe = self._read_log()
        self.preprocess()
    
    @cached_property
    def dataframe(self) -> pd.DataFrame:
        """Lazy-loaded DataFrame property."""
        self._ensure_data_loaded()
        return self._dataframe
    
    @property
    def file_path(self) -> Path:
        """Path to the event log file."""
        return self._file_path
    
    @file_path.setter
    def file_path(self, value: Union[str, Path]) -> None:
        self._file_path = Path(value)
        # Reset dataframe to force reload
        self._dataframe = None
    
    @property
    def unbiased_split_params(self) -> Dict[str, Any]:
        """Parameters for unbiased data splitting."""
        if self._unbiased_split_params is None:
            raise ValueError(
                f"Unbiased split not available for {self.__class__.__name__}"
            )
        return self._unbiased_split_params
    
    def __len__(self) -> int:
        """Number of events in the log."""
        return len(self.dataframe)
    
    def _download(self) -> None:
        """Download and prepare the event log file."""
        if not self.url or not self.file_name:
            raise ValueError("URL and file_name must be set for download")
        
        destination_folder = self.file_manager.root_folder / self.__class__.__name__
        
        try:
            downloaded_path = self.file_manager.download_and_extract(
                self.url, self.file_name, destination_folder
            )
            self._file_path = downloaded_path
            
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            raise
    
    def _read_log(self) -> pd.DataFrame:
        """Read event log from file with format detection."""
        file_path = self._file_path
        
        try:
            if file_path.suffix == ".xes":
                log = read_xes(str(file_path))
                
                if self.save_as_pandas:
                    # Convert and save as pandas format
                    new_path = file_path.with_suffix(self.config.default_file_format)
                    
                    if self.config.default_file_format == ".parquet":
                        log.to_parquet(new_path, index=False)
                    else:
                        raise ValueError(f"Unsupported format: {self.config.default_file_format}")
                    
                    # Update file path and remove old file
                    file_path.unlink()
                    self._file_path = new_path
                
                return log
            
            elif file_path.suffix == ".parquet":
                return pd.read_parquet(file_path)
            
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            self.logger.error(f"Failed to read log from {file_path}: {e}")
            raise
