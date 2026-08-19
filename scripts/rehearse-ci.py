#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml>=6"]
# ///
"""Rehearse every GitHub Actions workflow locally, before pushing.

    ./scripts/rehearse-ci.py [git-ref]        # defaults to HEAD

A plain ``uv run pytest`` in your working tree can pass while CI fails, because
a runner differs from your shell in ways this script reproduces:

* it starts from an **empty workspace** and populates it only if the job runs
  ``actions/checkout`` — so gitignored caches and files you forgot to ``git add``
  are invisible, and a job with no checkout (see the wheel-import job in
  ``package.yml``) really has no source tree to fall back on;
* every **matrix cell** runs, not just your default Python;
* each job gets a **fresh workspace**, so state cannot leak between jobs;
* artifacts move between jobs through an **artifact store**, never the disk.

The workflows, jobs, matrices and step order are all read from the YAML, so this
rehearses whatever is in ``.github/workflows`` today and picks up new workflows
with no changes here.

Workflow *definitions* are read from the working tree, so a workflow can be
rehearsed before it is committed; the *content* each job operates on comes from
the given ref. Uncommitted source changes are therefore not exercised, and the
script says so.

What it deliberately does *not* do: emulate a runner image, evaluate ``if:``
conditions, or run non-Linux matrix cells. An action it does not know how to
emulate is a hard failure rather than a silent skip — a rehearsal that quietly
stops covering a job is worse than no rehearsal at all.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import yaml

#: Actions emulated below. Anything else aborts the job it appears in.
_CHECKOUT = "actions/checkout"
_SETUP_UV = "astral-sh/setup-uv"
_UPLOAD = "actions/upload-artifact"
_DOWNLOAD = "actions/download-artifact"

#: Actions that only make sense against the real service, skipped by name so
#: that *unknown* actions can still be treated as an error.
_IGNORED = {"codecov/codecov-action"}

_EXPRESSION = re.compile(r"\$\{\{(.+?)\}\}")


class Unsupported(Exception):
    """A workflow used something this rehearsal cannot honestly emulate."""


@dataclass
class Outcome:
    workflow: str
    job: str
    cell: str
    status: str  # "pass" | "fail" | "skip"
    detail: str = ""

    @property
    def label(self) -> str:
        suffix = f" [{self.cell}]" if self.cell else ""
        return f"{self.workflow} / {self.job}{suffix}"


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, **kwargs)


def capture(command: list[str]) -> str:
    return run(command, stdout=subprocess.PIPE).stdout.strip()


def substitute(
    value,
    matrix: dict[str, str],
    steps: dict[str, dict[str, str]] | None = None,
    needs: dict[str, dict[str, str]] | None = None,
):
    """Resolve the ``${{ }}`` expressions GitHub would expand before bash runs.

    Supports the matrix, ``steps.<id>.outputs.<key>`` and
    ``needs.<job>.outputs.<key>``. Anything else is a hard error, so an
    unresolved expression can never be silently passed through to the shell.
    """
    if isinstance(value, (int, float, bool)) or value is None:
        return value

    def lookup(store: dict[str, dict[str, str]], expression: str, kind: str) -> str:
        owner, _, key = expression.partition(".outputs.")
        if not key:
            raise Unsupported(f"cannot evaluate expression: {kind}.{expression}")
        if owner not in store or key not in store[owner]:
            raise Unsupported(f"{kind}.{owner}.outputs.{key} was never set")
        return store[owner][key]

    def replace(match: re.Match) -> str:
        expression = match.group(1).strip()
        if expression.startswith("matrix."):
            key = expression.removeprefix("matrix.")
            if key not in matrix:
                raise Unsupported(f"matrix.{key} is not defined")
            return str(matrix[key])
        if expression.startswith("steps."):
            return lookup(steps or {}, expression.removeprefix("steps."), "steps")
        if expression.startswith("needs."):
            return lookup(needs or {}, expression.removeprefix("needs."), "needs")
        raise Unsupported(f"cannot evaluate expression: {expression}")

    return _EXPRESSION.sub(replace, str(value))


def read_key_values(path: Path) -> dict[str, str]:
    """Parse a ``$GITHUB_OUTPUT``/``$GITHUB_ENV`` file (``key=value`` lines)."""
    if not path.is_file():
        return {}
    pairs = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            pairs[key.strip()] = value
    return pairs


def matrix_cells(job: dict) -> list[dict[str, str]]:
    """Expand ``strategy.matrix`` into one dict per cell (GitHub's cross product)."""
    matrix = job.get("strategy", {}).get("matrix")
    if not matrix:
        return [{}]
    if unsupported := {"include", "exclude"} & set(matrix):
        raise Unsupported(f"matrix {'/'.join(sorted(unsupported))} is not supported")
    keys = list(matrix)
    return [dict(zip(keys, values)) for values in product(*(matrix[k] for k in keys))]


def order_jobs(jobs: dict) -> list[str]:
    """Topological order over ``needs:``, so a job runs after what it depends on."""
    remaining = {
        name: {n for n in _as_list(body.get("needs", []))} for name, body in jobs.items()
    }
    ordered: list[str] = []
    while remaining:
        ready = sorted(name for name, needs in remaining.items() if not needs - set(ordered))
        if not ready:
            raise Unsupported(f"cyclic needs: among {sorted(remaining)}")
        for name in ready:
            ordered.append(name)
            del remaining[name]
    return ordered


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    return [value] if value else []


def action_name(uses: str) -> str:
    return uses.split("@", 1)[0]


def rehearse_job(
    *,
    workflow: str,
    name: str,
    job: dict,
    cell: dict[str, str],
    source: Path,
    workspace: Path,
    artifacts: Path,
    needs: dict[str, dict[str, str]],
    step_outputs: dict[str, dict[str, str]],
) -> Outcome:
    label = ", ".join(f"{k}={v}" for k, v in cell.items())

    # A job-level condition decides whether GitHub runs the job at all. Rather
    # than guess at an event that has not happened, skip it and say why. This is
    # what keeps release.yml honest: its publish jobs are all guarded, so the
    # rehearsal covers the build and refuses to imply it covered an upload.
    if "if" in job:
        return Outcome(workflow, name, label, "skip", f"job condition: {job['if']}")

    runner = str(substitute(job.get("runs-on", "ubuntu-latest"), cell))
    if not runner.startswith("ubuntu"):
        return Outcome(workflow, name, label, "skip", f"cannot run {runner} locally")

    print(f"\n\033[1m===== {workflow} / {name}" + (f" [{label}]" if label else "") + " =====\033[0m")
    workspace.mkdir(parents=True)

    # A runner starts with a clean environment; an inherited VIRTUAL_ENV would
    # let the project venv stand in for one the job never created.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update({k: str(v) for k, v in (job.get("env") or {}).items()})
    # The subset of the runner environment that steps legitimately read. The
    # event-shaped values are nominal: a rehearsal is not a real event.
    env.update(
        {
            "CI": "true",
            "GITHUB_WORKSPACE": str(workspace),
            "GITHUB_REF_NAME": capture(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "RUNNER_OS": "Linux",
        }
    )
    timeout = float(job.get("timeout-minutes", 360)) * 60

    # GitHub gives each step files to write outputs and env vars into. They live
    # outside the workspace so upload-artifact cannot pick them up.
    meta = workspace.parent / f"{workspace.name}-meta"
    meta.mkdir(parents=True, exist_ok=True)

    for step in job.get("steps", []):
        title = step.get("name") or step.get("uses") or "run"

        if "if" in step:
            print(f"  -- skipping {title!r}: has an if: condition")
            continue

        if uses := step.get("uses"):
            action = action_name(uses)
            with_ = {
                k: substitute(v, cell, step_outputs, needs)
                for k, v in (step.get("with") or {}).items()
            }

            if action == _CHECKOUT:
                print(f"  -- {title}: populating the workspace from {source.name}")
                shutil.copytree(source, workspace, dirs_exist_ok=True)
            elif action == _SETUP_UV:
                # setup-uv exports UV_PYTHON; the run steps rely on it rather
                # than passing -p, so the rehearsal has to set it too.
                if python := with_.get("python-version"):
                    env["UV_PYTHON"] = str(python)
                    print(f"  -- {title}: UV_PYTHON={python}")
            elif action == _UPLOAD:
                store = artifacts / str(with_["name"])
                store.mkdir(parents=True, exist_ok=True)
                shutil.copytree(workspace / str(with_["path"]), store, dirs_exist_ok=True)
                print(f"  -- {title}: stored artifact {with_['name']!r}")
            elif action == _DOWNLOAD:
                store = artifacts / str(with_["name"])
                if not store.is_dir():
                    return Outcome(workflow, name, label, "fail", f"no artifact {with_['name']!r}")
                target = workspace / str(with_.get("path", "."))
                shutil.copytree(store, target, dirs_exist_ok=True)
                print(f"  -- {title}: restored artifact {with_['name']!r}")
            elif action in _IGNORED:
                print(f"  -- skipping {title!r}: needs the real service")
            else:
                return Outcome(workflow, name, label, "fail", f"unsupported action: {action}")
            continue

        if "run" not in step:
            continue

        script = substitute(step["run"], cell, step_outputs, needs)
        cwd = workspace / str(step.get("working-directory", "."))
        print(f"  -- {title}")

        index = len(step_outputs)
        output_file = meta / f"output-{index}"
        env_file = meta / f"env-{index}"
        output_file.touch()
        env_file.touch()
        step_env = env | {
            "GITHUB_OUTPUT": str(output_file),
            "GITHUB_ENV": str(env_file),
            "GITHUB_STEP_SUMMARY": str(meta / f"summary-{index}"),
        }
        try:
            subprocess.run(
                ["bash", "-euo", "pipefail", "-c", script],
                cwd=cwd,
                env=step_env,
                timeout=timeout,
                check=True,
            )
        except subprocess.TimeoutExpired:
            return Outcome(workflow, name, label, "fail", f"exceeded timeout-minutes ({timeout / 60:g})")
        except subprocess.CalledProcessError as error:
            return Outcome(workflow, name, label, "fail", f"{title!r} exited {error.returncode}")

        # $GITHUB_OUTPUT becomes steps.<id>.outputs; $GITHUB_ENV persists into
        # the remaining steps, exactly as it does on a runner.
        step_outputs[str(step.get("id", f"#{index}"))] = read_key_values(output_file)
        env.update(read_key_values(env_file))

    return Outcome(workflow, name, label, "pass")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref", nargs="?", default="HEAD", help="git ref to rehearse")
    args = parser.parse_args()

    root = Path(capture(["git", "rev-parse", "--show-toplevel"]))
    os.chdir(root)

    workflow_files = sorted(
        path
        for pattern in ("*.yml", "*.yaml")
        for path in Path(".github/workflows").glob(pattern)
    )
    if not workflow_files:
        print("no workflows found under .github/workflows", file=sys.stderr)
        return 1

    # The rehearsal runs a committed ref, so uncommitted work is invisible to it.
    if capture(["git", "status", "--porcelain"]):
        print(
            "\033[33mwarning: working tree is dirty — workflow files are read from it, "
            "but uncommitted source changes are NOT rehearsed\033[0m"
        )

    print("===== actionlint =====")
    try:
        run(["uvx", "--from", "actionlint-py", "actionlint", *map(str, workflow_files)])
    except subprocess.CalledProcessError:
        return 1
    print("ok")

    revision = capture(["git", "rev-parse", "--short", args.ref])
    print(f"rehearsing {revision} ({len(workflow_files)} workflows)")

    outcomes: list[Outcome] = []
    with tempfile.TemporaryDirectory(prefix="rehearse-") as tmp:
        scratch = Path(tmp)

        # One pristine export of the committed tree; each job that checks out
        # gets its own copy, mirroring one fresh machine per job.
        source = scratch / "source"
        source.mkdir()
        tarball = scratch / "source.tar"
        with tarball.open("wb") as handle:
            subprocess.run(
                ["git", "archive", "--format=tar", args.ref], stdout=handle, check=True
            )
        shutil.unpack_archive(tarball, source, format="tar")

        for index, path in enumerate(workflow_files):
            document = yaml.safe_load(path.read_text())
            jobs = document.get("jobs") or {}
            name = document.get("name") or path.stem
            artifacts = scratch / f"artifacts-{index}"
            job_outputs: dict[str, dict[str, str]] = {}

            try:
                job_order = order_jobs(jobs)
            except Unsupported as error:
                outcomes.append(Outcome(name, "-", "", "fail", str(error)))
                continue

            for position, job_name in enumerate(job_order):
                job = jobs[job_name]
                try:
                    cells = matrix_cells(job)
                except Unsupported as error:
                    outcomes.append(Outcome(name, job_name, "", "fail", str(error)))
                    continue

                for cell_index, cell in enumerate(cells):
                    workspace = scratch / f"{index}-{position}-{cell_index}"
                    step_outputs: dict[str, dict[str, str]] = {}
                    try:
                        outcome = rehearse_job(
                            workflow=name,
                            name=job_name,
                            job=job,
                            cell=cell,
                            source=source,
                            workspace=workspace,
                            artifacts=artifacts,
                            needs=job_outputs,
                            step_outputs=step_outputs,
                        )
                        if outcome.status == "pass":
                            job_outputs[job_name] = {
                                key: str(substitute(value, cell, step_outputs, job_outputs))
                                for key, value in (job.get("outputs") or {}).items()
                            }
                        outcomes.append(outcome)
                    except Unsupported as error:
                        outcomes.append(Outcome(name, job_name, "", "fail", str(error)))

    print("\n\033[1m===== summary =====\033[0m")
    marks = {"pass": "\033[32mPASS\033[0m", "fail": "\033[31mFAIL\033[0m", "skip": "\033[33mSKIP\033[0m"}
    for outcome in outcomes:
        detail = f"  ({outcome.detail})" if outcome.detail else ""
        print(f"  {marks[outcome.status]}  {outcome.label}{detail}")

    failures = [o for o in outcomes if o.status == "fail"]
    print(f"\n{len(failures)} failed, {sum(o.status == 'pass' for o in outcomes)} passed, "
          f"{sum(o.status == 'skip' for o in outcomes)} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
