"""Cache-folder creation for the 4TU loaders.

These tests exercise the real directory-creation code in
:class:`skpm.event_logs.base.TUEventLog`. Only the network download is
substituted (by a subclass that writes a tiny CSV locally), so the folder
handling under test is the production code path.
"""

from pathlib import Path

import pandas as pd
import pytest

from skpm.event_logs.base import TUEventLog


class LocalCsvLog(TUEventLog):
    """A 4TU-style loader whose download step writes a local CSV.

    Stands in for e.g. ``BPI17`` so the cache-folder logic can be tested
    without a 1.6M-event download.
    """

    url = ("local://no-network",)
    file_name = ("log.csv",)

    def _download(self) -> None:
        pd.DataFrame(
            {
                "case:concept:name": ["1", "1", "2"],
                "time:timestamp": [
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:00:00",
                    "2020-01-02 00:00:00",
                ],
                "concept:name": ["a", "b", "a"],
            }
        ).to_csv(self.file_path[0], index=False)


def test_default_cache_folder_is_created_when_home_has_no_skpm_dir(
    tmp_path, monkeypatch
):
    """A fresh machine has no ``~/skpm/event_logs``; the loader must create it.

    Regression test: ``mkdir(exist_ok=True)`` without ``parents=True`` raised
    ``FileNotFoundError`` for every loader on a machine that had never run
    skpm before.
    """
    fresh_home = tmp_path / "fresh-home"
    fresh_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fresh_home))

    log = LocalCsvLog()

    assert (
        log.cache_folder == fresh_home / "skpm" / "event_logs" / "LocalCsvLog"
    )
    assert log.cache_folder.is_dir()
    assert isinstance(log.dataframe, pd.DataFrame)
    assert len(log.dataframe) == 3


def test_per_log_subfolder_is_created_under_a_user_cache_folder(tmp_path):
    """The ``<cache_folder>/<ClassName>`` subfolder is created on demand."""
    log = LocalCsvLog(cache_folder=tmp_path)

    assert log.cache_folder == tmp_path / "LocalCsvLog"
    assert log.cache_folder.is_dir()
    assert len(log.dataframe) == 3


def test_cache_folder_pointing_at_a_file_raises(tmp_path):
    """A user-supplied cache_folder must be a directory, not a file."""
    not_a_dir = tmp_path / "some.parquet"
    not_a_dir.touch()

    with pytest.raises(ValueError, match="must be a directory"):
        LocalCsvLog(cache_folder=not_a_dir)
