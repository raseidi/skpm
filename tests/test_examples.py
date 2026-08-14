"""Execute the gallery examples that are safe to run in CI.

Every ``examples/plot_*.py`` file *executes* during the docs build (see
``filename_pattern`` in ``docs/source/conf.py``), so a broken example fails the
Read the Docs build. Running them here surfaces the breakage in CI instead,
where it is cheap to diagnose.

Examples that reach the network are excluded — the 4TU repository being slow or
down must not fail the unit suite. Keep :data:`_NEEDS_NETWORK` in step with the
examples that construct a loader.
"""

import runpy
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

#: Examples the docs build runs but the test suite must not: these construct a
#: loader, which downloads from the 4TU repository on a cold cache.
_NEEDS_NETWORK = {"plot_download_event_logs.py"}


def _runnable_examples() -> list[Path]:
    return sorted(
        path
        for path in EXAMPLES_DIR.glob("plot_*.py")
        if path.name not in _NEEDS_NETWORK
    )


def test_there_is_at_least_one_runnable_example():
    """Guard the guard: a bad glob would make the test below vacuously pass."""
    assert _runnable_examples()


@pytest.mark.parametrize(
    "example", _runnable_examples(), ids=lambda path: path.name
)
def test_example_runs_without_error(example, monkeypatch):
    """The example executes top to bottom, as the docs build will run it."""
    # Headless: the docs build uses the same non-interactive backend.
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.close("all")
    try:
        runpy.run_path(str(example), run_name="__main__")
    finally:
        plt.close("all")


def test_quickstart_produces_a_figure_for_the_gallery():
    """The gallery thumbnail comes from an open figure; no figure, no thumbnail."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.close("all")
    try:
        runpy.run_path(str(EXAMPLES_DIR / "plot_quickstart.py"))
        assert plt.get_fignums(), "plot_quickstart.py left no figure open"
    finally:
        plt.close("all")
