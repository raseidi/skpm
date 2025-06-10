from typing import Dict, Optional

class _EventLogConfig:
    """
    Global configuration manager for event log column mappings and file formats.

    This class supports global configuration following the classic XES naming.
    Changes to the global config  automatically propagate to all instances.
    SkPM will never rename columns, so users should ensure their data matches the
    expected column names.
    
    Attributes
    ----------
    case_id : str
        Case ID column name.
    activity : str
        Activity column name.
    timestamp : str
        Timestamp column name.
    resource : str
        Resource column name (optional, not used in this class but can be added).

    Methods
    -------
    to_dict() -> Dict[str, str]
        Convert current configuration to dictionary.
    get_global_config() -> Dict[str, str]
        Get the current global configuration.
    set_global_config(case_id: Optional[str] = None, activity: Optional[str] = None, timestamp: Optional[str] = None) -> None
        Set the global configuration.
    reset_global_config() -> None
        Reset global configuration to defaults.

    Examples
    --------
    # Set global configuration
    EventLogConfig.set_global_config(case_id='case_id', activity='activity')

    # Reset global configuration to defaults
    EventLogConfig.reset_global_config()
    
    Notes
    -----
    The global configuration is shared across all instances of EventLogConfig.
    If you need instance-specific configurations, consider subclassing EventLogConfig.
    This class is designed to be used as a singleton, where the global configuration
    is modified through class methods rather than instance methods.
    """

    # Global default configuration - shared across all instances
    _GLOBAL_CONFIG = {
        "case_id": "case:concept:name",
        "activity": "concept:name",
        "timestamp": "time:timestamp",
        "resource": "org:resource",  # should be optional
        "EOT": "<EOT>",  # End of trace marker
    }
    _instance = None

    def __new__(cls):
        """Ensure only one instance of EventLogConfig exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

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
    
    @property
    def resource(self) -> str:
        """Resource column name."""
        return self._GLOBAL_CONFIG["resource"]
    
    @property
    def EOT(self) -> str:
        """End of trace marker."""
        return self._GLOBAL_CONFIG["EOT"]

    def to_dict(self) -> Dict[str, str]:
        """Convert current configuration to dictionary."""
        return {
            "case_id": self.case_id,
            "activity": self.activity,
            "timestamp": self.timestamp,
            "resource": self.resource,
            "EOT": self.EOT,
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
        EOT: Optional[str] = None,
    ) -> None:
        """Set the global configuration."""
        if case_id is not None:
            cls._GLOBAL_CONFIG["case_id"] = case_id
        if activity is not None:
            cls._GLOBAL_CONFIG["activity"] = activity
        if timestamp is not None:
            cls._GLOBAL_CONFIG["timestamp"] = timestamp
        if resource is not None:
            cls._GLOBAL_CONFIG["resource"] = resource
        if EOT is not None:
            cls._GLOBAL_CONFIG["EOT"] = EOT

    @classmethod
    def reset_global_config(cls) -> None:
        """Reset global configuration to defaults."""
        cls._GLOBAL_CONFIG = {
            "case_id": "case:concept:name",
            "activity": "concept:name",
            "timestamp": "time:timestamp",
            "resource": "org:resource",
            "EOT": "<EOT>",
        }

    def __repr__(self) -> str:
        """String representation showing current configuration."""
        config = self.to_dict()

        lines = ["EventLogConfig("]
        for key, value in config.items():
            lines.append(f"  {key}='{value}'")

        return "\n".join(lines) + "\n)"

EventLogConfig: _EventLogConfig = _EventLogConfig()

class EventLogConfigMixin:
    """Mixin class to provide event log configuration properties.
    
    This mixin will be used globally across SkPM to ensure that
    all event log handling classes have access to the same
    configuration properties for case ID, activity, and timestamp.
    
    *Note*: SkPM will never rename columns, so users should ensure their data matches the
    expected column names.
    Attributes
    ----------
    case_id : str
        Case ID column name.
    activity : str
        Activity column name.
    timestamp : str
        Timestamp column name.
    """

    _config: _EventLogConfig = _EventLogConfig()

    @property
    def case_id(self) -> str:
        return self._config.case_id

    @property
    def activity(self) -> str:
        return self._config.activity

    @property
    def timestamp(self) -> str:
        return self._config.timestamp