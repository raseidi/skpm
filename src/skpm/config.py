from typing import Dict, Final, Mapping, Optional, Tuple

import pandas as pd

__all__ = ["EventLogConfig", "EventLogConfigMixin"]

# Fixed canonical field names — how skpm addresses an event log's semantic
# columns internally, once loaded. NOT configurable. (case_id + timestamp
# become index levels; activity + resource stay as columns.)
CANONICAL_FIELDS: Final[Tuple[str, ...]] = (
    "case_id",
    "timestamp",
    "activity",
    "resource",
)

# An event log is defined by case_id + timestamp + activity; resource is
# optional.
REQUIRED_FIELDS: Final[Tuple[str, ...]] = ("case_id", "timestamp", "activity")

# Default *source* column names skpm expects in raw input (the XES standard).
# These are the single expected name per field — not a set of aliases. Users
# override them via set_global_config(...) or the per-call column_mapping.
_XES_SOURCE_NAMES: Final[Dict[str, str]] = {
    "case_id": "case:concept:name",
    "timestamp": "time:timestamp",
    "activity": "concept:name",
    "resource": "org:resource",
}


class _EventLogConfig:
    """Declares how an event log's source columns are named, and maps them to
    skpm's fixed canonical names at load.

    Two distinct notions:

    * **Canonical names** (:data:`CANONICAL_FIELDS`) — ``case_id``,
      ``timestamp``, ``activity``, ``resource`` — are constants and are how
      skpm addresses columns internally. Exposed as the ``.case_id`` etc.
      properties.
    * **Source names** — the column names skpm expects in *raw* input, one per
      field, defaulting to the XES standard. This is what's configurable.

    Declare your source naming globally with :meth:`set_global_config` (e.g.
    ``set_global_config(case_id="CaseID")`` if your logs call it ``CaseID``),
    or per call via the ``mapping`` argument of :meth:`normalize_columns`.
    There is no alias guessing: a field resolves only from an explicit
    mapping, the configured source name, or an already-canonical column.
    """

    __slots__ = ("_source_names",)

    def __init__(self) -> None:
        self._source_names: Dict[str, str] = dict(_XES_SOURCE_NAMES)

    # -- fixed canonical names (how skpm addresses columns internally) --------
    @property
    def case_id(self) -> str:
        return "case_id"

    @property
    def timestamp(self) -> str:
        return "timestamp"

    @property
    def activity(self) -> str:
        return "activity"

    @property
    def resource(self) -> str:
        return "resource"

    # -- source-name configuration -------------------------------------------
    def set_global_config(
        self,
        case_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        activity: Optional[str] = None,
        resource: Optional[str] = None,
    ) -> None:
        """Declare the source column names used in your event logs.

        Each argument sets the expected *source* name for that field,
        replacing the default. e.g. ``set_global_config(case_id="CaseID")``
        tells skpm your case-id column is named ``CaseID``. The canonical
        output names are unaffected.
        """
        for field, name in (
            ("case_id", case_id),
            ("timestamp", timestamp),
            ("activity", activity),
            ("resource", resource),
        ):
            if name is not None:
                self._source_names[field] = name

    def reset_global_config(self) -> None:
        """Reset the expected source names to the XES defaults."""
        self._source_names = dict(_XES_SOURCE_NAMES)

    def get_global_config(self) -> Dict[str, str]:
        """Return the configured source column names (field -> source name)."""
        return dict(self._source_names)

    def to_dict(self) -> Dict[str, str]:
        """Return the fixed canonical names (field -> canonical name)."""
        return {field: getattr(self, field) for field in CANONICAL_FIELDS}

    # -- resolution -----------------------------------------------------------
    def _resolve_field(
        self, field: str, columns: list, mapping: Mapping[str, str]
    ) -> Optional[str]:
        """Return the source column in ``columns`` that maps to ``field``."""
        # 1. explicit mapping wins
        if field in mapping:
            src = mapping[field]
            if src not in columns:
                raise ValueError(
                    f"column_mapping[{field!r}] = {src!r}, which is not a "
                    f"column of the DataFrame ({columns})."
                )
            return src
        # 2. configured source name present
        configured = self._source_names[field]
        if configured in columns:
            return configured
        # 3. already canonical
        if field in columns:
            return field
        return None

    def normalize_columns(
        self,
        df: pd.DataFrame,
        mapping: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        """Rename an event log's source columns to the fixed canonical names.

        For each field the source column is resolved by precedence — explicit
        ``mapping`` > configured source name > already-canonical column — then
        renamed to the canonical name. Required fields (``case_id``,
        ``timestamp``, ``activity``) raise if unresolved; ``resource`` is
        optional. The input is not mutated.

        Raises
        ------
        ValueError
            If a required field cannot be resolved, or an explicit mapping
            points at a column that is not present.
        """
        mapping = dict(mapping or {})
        columns = list(df.columns)
        rename_map: Dict[str, str] = {}
        for field in CANONICAL_FIELDS:
            src = self._resolve_field(field, columns, mapping)
            if src is None:
                if field in REQUIRED_FIELDS:
                    raise ValueError(
                        f"Cannot resolve required column {field!r}: expected "
                        f"source name {self._source_names[field]!r} (configure via "
                        f"set_global_config or pass column_mapping={{'{field}': "
                        f"'<column>'}}) is not among {columns}."
                    )
                continue  # optional field absent
            if src != field:
                rename_map[src] = field

        return df.rename(columns=rename_map) if rename_map else df

    def __repr__(self) -> str:
        lines = ["EventLogConfig(canonical <- source)"]
        for field in CANONICAL_FIELDS:
            req = "required" if field in REQUIRED_FIELDS else "optional"
            lines.append(f"  {field} ({req}) <- {self._source_names[field]!r}")
        return "\n".join(lines)


EventLogConfig: _EventLogConfig = _EventLogConfig()


class EventLogConfigMixin:
    """Mixin exposing the canonical event-log field names to estimators.

    The names (``case_id``, ``timestamp``, ``activity``, ``resource``) are
    fixed constants supplied by :data:`EventLogConfig`. They are how estimators
    address the index levels (``case_id``/``timestamp``) and the activity /
    resource columns of a canonical event log. Because the names are constant
    there is no fit-time state to snapshot.
    """

    _config: _EventLogConfig = EventLogConfig

    @property
    def case_id(self) -> str:
        return self._config.case_id

    @property
    def timestamp(self) -> str:
        return self._config.timestamp

    @property
    def activity(self) -> str:
        return self._config.activity

    @property
    def resource(self) -> str:
        return self._config.resource
