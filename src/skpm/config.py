from typing import Dict, Final, Mapping, Optional, Tuple

import pandas as pd

__all__ = ["EventLogConfig", "EventLogConfigMixin"]


_XES_DEFAULTS: Final[Dict[str, str]] = {
    "case_id": "case:concept:name",
    "activity": "concept:name",
    "timestamp": "time:timestamp",
    "resource": "org:resource",
}

_REQUIRED_KEYS: Final[Tuple[str, ...]] = ("case_id", "activity", "timestamp")
_RENAMEABLE_KEYS: Final[Tuple[str, ...]] = (
    "case_id",
    "activity",
    "timestamp",
    "resource",
)


class _EventLogConfig:
    """Global configuration for event log column names.

    A single module-level instance, :data:`EventLogConfig`, holds the mapping
    between semantic keys (``case_id``, ``activity``, ``timestamp``,
    ``resource``) and the DataFrame column names SkPM expects.
    Defaults follow the XES standard.

    Users can either:

    1. Reconfigure the global standard column names via
       :meth:`set_global_config`, so SkPM looks for *those* names in event
       logs; or
    2. Provide a source-to-standard mapping at load time via
       :meth:`normalize_columns`, which renames source columns to the
       configured standard names.

    The configuration is shared across the process; estimators read it
    through :class:`EventLogConfigMixin`.
    """

    __slots__ = ("_state",)

    def __init__(self) -> None:
        self._state: Dict[str, str] = dict(_XES_DEFAULTS)

    @property
    def case_id(self) -> str:
        return self._state["case_id"]

    @property
    def activity(self) -> str:
        return self._state["activity"]

    @property
    def timestamp(self) -> str:
        return self._state["timestamp"]

    @property
    def resource(self) -> str:
        return self._state["resource"]

    def to_dict(self) -> Dict[str, str]:
        """Return a copy of the current configuration."""
        return dict(self._state)

    def get_global_config(self) -> Dict[str, str]:
        """Return a copy of the current global configuration."""
        return dict(self._state)

    def set_global_config(
        self,
        case_id: Optional[str] = None,
        activity: Optional[str] = None,
        timestamp: Optional[str] = None,
        resource: Optional[str] = None,
    ) -> None:
        """Update one or more standard column names."""
        updates = {
            "case_id": case_id,
            "activity": activity,
            "timestamp": timestamp,
            "resource": resource,
        }
        for key, value in updates.items():
            if value is not None:
                self._state[key] = value

    def reset_global_config(self) -> None:
        """Reset the global configuration to XES defaults."""
        self._state.clear()
        self._state.update(_XES_DEFAULTS)

    def normalize_columns(
        self,
        df: pd.DataFrame,
        mapping: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        """Rename source columns to the configured standard names.

        For each semantic key (``case_id``, ``activity``, ``timestamp``,
        ``resource``), if the configured standard name is already present
        in ``df``, the column is left untouched. Otherwise, if ``mapping``
        provides a source column name for that key and it exists in
        ``df``, that column is renamed to the standard name.

        Parameters
        ----------
        df : pandas.DataFrame
            Event log to normalize.
        mapping : Mapping[str, str], optional
            Mapping from semantic key (``"case_id"``, ``"activity"``,
            ``"timestamp"``, ``"resource"``) to the source column name in
            ``df``.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with standardized column names. The input is not
            mutated.

        Raises
        ------
        ValueError
            If a required column (``case_id``, ``activity``,
            ``timestamp``) cannot be resolved from either the standard
            name or the provided mapping.
        """
        mapping = dict(mapping or {})
        rename_map: Dict[str, str] = {}
        for key in _RENAMEABLE_KEYS:
            standard = self._state[key]
            if standard in df.columns:
                continue
            source = mapping.get(key)
            if source is not None and source in df.columns:
                rename_map[source] = standard
            elif key in _REQUIRED_KEYS:
                raise ValueError(
                    f"Cannot resolve required column for '{key}': "
                    f"standard name {standard!r} not present in DataFrame "
                    f"and no valid source column provided in `mapping`."
                )

        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def __repr__(self) -> str:
        lines = ["EventLogConfig("]
        for key, value in self._state.items():
            lines.append(f"  {key}='{value}'")
        return "\n".join(lines) + "\n)"


EventLogConfig: _EventLogConfig = _EventLogConfig()


class EventLogConfigMixin:
    """Mixin providing event-log column-name properties.

    Before fit, the properties (``case_id``, ``activity``, ``timestamp``,
    ``resource``) read live values from the shared
    :data:`EventLogConfig` singleton, so estimators pick up the user's
    current configuration as a sensible default.

    At fit time, :meth:`_snapshot_config` copies the current values into
    trailing-underscore attributes (``case_id_``, ``activity_``,
    ``timestamp_``, ``resource_``,). From that point on, the
    properties return the snapshot, mirroring scikit-learn's convention
    that fitted estimators expose their learned state via
    ``trailing_underscore_`` attributes (``feature_names_in_``,
    ``classes_``, ...).

    The consequence is that a fitted estimator is self-contained: later
    mutations to the global :data:`EventLogConfig` do not change its
    behavior, and pickling preserves the column-name contract it was
    trained under. Each step in a scikit-learn :class:`Pipeline`
    snapshots independently when its ``fit`` runs.
    """

    _config: _EventLogConfig = EventLogConfig

    @property
    def case_id(self) -> str:
        return self.case_id_ if hasattr(self, "case_id_") else self._config.case_id

    @property
    def activity(self) -> str:
        return self.activity_ if hasattr(self, "activity_") else self._config.activity

    @property
    def timestamp(self) -> str:
        return (
            self.timestamp_
            if hasattr(self, "timestamp_")
            else self._config.timestamp
        )

    @property
    def resource(self) -> str:
        return self.resource_ if hasattr(self, "resource_") else self._config.resource

    def _snapshot_config(self) -> None:
        """Freeze the current global column-name config onto this estimator.

        Sets ``case_id_``, ``activity_``, ``timestamp_``, ``resource_``,
        and from :data:`EventLogConfig`. Call this from ``fit``
        before any computation so the rest of fit/transform reads a
        consistent snapshot, regardless of later changes to the global
        config.
        """
        cfg = self._config
        self.case_id_ = cfg.case_id
        self.activity_ = cfg.activity
        self.timestamp_ = cfg.timestamp
        self.resource_ = cfg.resource
