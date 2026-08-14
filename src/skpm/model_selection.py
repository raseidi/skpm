"""Splitting event logs into training and test sets.

Mirrors :mod:`sklearn.model_selection`: :func:`train_test_split` is the
documented door, and the two strategies it dispatches to
(:func:`~skpm.event_logs.split.temporal`,
:func:`~skpm.event_logs.split.unbiased`) stay importable as primitives.

Splitting comes **first** in a skpm workflow — before feature extraction,
before fitting anything::

    from skpm.event_logs import BPI17
    from skpm.model_selection import train_test_split

    train, test = train_test_split(BPI17(), strategy="unbiased")

The split returns two event logs, not ``(X_train, X_test, y_train, y_test)``.
Which target to predict is a modelling decision — remaining time is a
regression, next activity a classification — so it is made after the split,
per side::

    from skpm.feature_extraction.targets import remaining_time

    y_train = remaining_time(train, time_unit="h")
    y_test = remaining_time(test, time_unit="h")

Both strategies are **case-level**: every case lands entirely on one side.
That is what keeps the targets well defined on each side, and it means
computing a target before or after the split gives identical values.
"""

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import pandas as pd

from skpm.event_logs.base import LogLike, TUEventLog, to_event_log
from skpm.event_logs.split import temporal, unbiased

__all__ = ["train_test_split", "temporal", "unbiased"]

#: Strategy name -> the primitive implementing it.
_STRATEGIES = {"temporal": temporal, "unbiased": unbiased}


def _loader_params(log: LogLike) -> Dict[str, Any]:
    """The unbiased parameters a loader ships, or an empty dict.

    Must run *before* :func:`~skpm.event_logs.base.to_event_log`, which erases
    the distinction between a loader and its frame — this is the one place in
    skpm that legitimately inspects the input's type outside that boundary.

    Narrowed to :class:`~skpm.event_logs.base.TUEventLog` because that is where
    the published constants live; a bare ``EventLog`` has no such attribute.
    """
    if not isinstance(log, TUEventLog):
        return {}
    try:
        return log.unbiased_split_params
    except ValueError:
        # Loaders that ship no parameters raise; that is not an error here, it
        # just means there is nothing to resolve from and the caller must pass
        # them (or get the error message below).
        return {}


def train_test_split(
    log: LogLike,
    *,
    strategy: str = "temporal",
    test_size: float = 0.2,
    start_date: Optional[Union[str, pd.Period]] = None,
    end_date: Optional[Union[str, pd.Period]] = None,
    max_days: Optional[float] = None,
    column_mapping: Optional[Mapping[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split an event log into training and test sets, by whole cases.

    Parameters
    ----------
    log : EventLog or pandas.DataFrame
        An :class:`~skpm.event_logs.base.EventLog` (any loader in
        :mod:`skpm.event_logs`), a flat DataFrame, or an already-canonical
        DataFrame. Coercion is delegated to
        :func:`~skpm.event_logs.base.to_event_log`.
    strategy : {"temporal", "unbiased"}, default="temporal"
        ``"temporal"`` cuts on a timestamp derived from ``test_size`` and
        needs no other parameters, so it works on any log. ``"unbiased"``
        additionally drops cases that the recording window truncates,
        following [1], and needs the three parameters below.
    test_size : float, default=0.2
        Size of the test side, as a fraction strictly between 0 and 1. Its
        meaning follows the strategy: for ``"unbiased"`` it is a proportion of
        **cases**; for ``"temporal"`` it is a fraction of the log's **calendar
        span**, so uneven case arrivals give uneven case counts.
    start_date, end_date : str or pandas.Period, optional
        ``strategy="unbiased"`` only. Bounds on case start/end, at monthly
        resolution.
    max_days : float, optional
        ``strategy="unbiased"`` only. Maximum allowed case duration in days.
    column_mapping : Mapping[str, str], optional
        Semantic key (``"case_id"``, ``"timestamp"``, ``"activity"``,
        ``"resource"``) to source column name, for a flat DataFrame whose
        columns follow neither the XES nor the canonical naming. Passing it
        with an :class:`~skpm.event_logs.base.EventLog` raises, since a loader
        normalized its columns on construction.

    Notes
    -----
    When ``strategy="unbiased"`` and ``log`` is a loader that publishes
    :attr:`~skpm.event_logs.base.TUEventLog.unbiased_split_params`, those
    parameters fill in whichever of the three you leave unset; anything you
    pass explicitly wins. A loader is the only thing that carries them, so
    for a plain DataFrame they are required.

    Returns
    -------
    (train, test) : tuple of pandas.DataFrame
        Two event logs in canonical form, each carrying the
        ``MultiIndex(case_id, timestamp, event_id)``. No target is returned:
        build it per side with
        :mod:`skpm.feature_extraction.targets`.

    Raises
    ------
    ValueError
        If ``strategy`` is unknown; if ``test_size`` is not strictly between
        0 and 1; if ``strategy="unbiased"`` and ``max_days`` can be resolved
        from neither the arguments nor the loader; if an unbiased-only
        parameter is passed with ``strategy="temporal"``; if the unbiased
        filters leave no cases; or if the split leaves either side empty.

    Examples
    --------
    >>> from skpm.event_logs import BPI17
    >>> from skpm.model_selection import train_test_split
    >>> train, test = train_test_split(BPI17(), strategy="unbiased")

    A log of your own needs no loader class, and no ``to_event_log`` call:

    >>> train, test = train_test_split(pd.read_csv("mine.csv"))  # doctest: +SKIP

    References
    ----------
    .. [1] Hans Weytjens, Jochen De Weerdt. Creating Unbiased Public
       Benchmark Datasets with Data Leakage Prevention for Predictive
       Process Monitoring, 2021. doi:10.1007/978-3-030-94343-1_2
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Valid strategies are: "
            f"{', '.join(sorted(_STRATEGIES))}."
        )

    given = {
        "start_date": start_date,
        "end_date": end_date,
        "max_days": max_days,
    }

    # Resolve loader-published constants before coercing: to_event_log erases
    # the difference between a loader and its frame.
    resolved = dict(_loader_params(log))
    log = to_event_log(log, column_mapping=column_mapping)

    if strategy == "temporal":
        passed = [name for name, value in given.items() if value is not None]
        if passed:
            raise ValueError(
                f"{', '.join(passed)} only applies to strategy='unbiased', "
                f"but strategy='temporal' was given. Either pass "
                f"strategy='unbiased', or drop {', '.join(passed)}."
            )
        return temporal(log, test_size=test_size)

    resolved.update(
        {name: value for name, value in given.items() if value is not None}
    )

    if resolved.get("max_days") is None:
        raise ValueError(
            "strategy='unbiased' needs max_days (and optionally start_date / "
            "end_date). They are per-dataset constants published with the "
            "benchmark, so they are read automatically from a loader that "
            "ships them — pass them explicitly for any other log, e.g. "
            "train_test_split(log, strategy='unbiased', max_days=47.81)."
        )

    return unbiased(
        log,
        start_date=resolved.get("start_date"),
        end_date=resolved.get("end_date"),
        max_days=resolved["max_days"],
        test_size=test_size,
    )
