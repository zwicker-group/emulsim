"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os
import subprocess as sp
import sys
from pathlib import Path

import pytest

from pde.tools.misc import module_available

# get path of the package and the example directory
PACKAGEPATH = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = PACKAGEPATH / "examples"

# create set of examples that need to be skipped
SKIP_EXAMPLES: set[str] = set()
if not module_available("phasesep"):
    SKIP_EXAMPLES.add("droplets_active.py")
if not module_available("napari"):
    SKIP_EXAMPLES.add("droplets_interactive.py")
if not module_available("numba_scipy"):
    SKIP_EXAMPLES.add("droplets_active.py")
    SKIP_EXAMPLES.add("droplets_trackers.py")

# get dictionary of files that need to be deleted after the examples ran
CLEANUP: dict[str, Path] = {"storing_data.py": PACKAGEPATH / "trajectory.zip"}


@pytest.mark.no_cover
@pytest.mark.skipif(sys.platform == "win32", reason="Assumes unix setup")
@pytest.mark.parametrize("path", EXAMPLE_PATH.glob("**/*.py"))
def test_example(path):
    """Runs an example script given by path."""
    # check whether the example should actually by ran
    if ".ipynb_checkpoints" in str(path):
        pytest.skip("Python file in private directory")
    if path.name.startswith("_"):
        pytest.skip("skip examples starting with an underscore")
    if any(name in str(path) for name in SKIP_EXAMPLES):
        pytest.skip(f"Skip test {path}")

    # run the example script
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGEPATH) + ":" + env.get("PYTHONPATH", "")
    proc = sp.Popen([sys.executable, path], env=env, stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        outs, errs = proc.communicate(timeout=30)
    except sp.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()

    # clean up potential files that were written
    for script, delete_path in CLEANUP.items():
        if script in str(path):
            delete_path.unlink(missing_ok=True)

    # check its output
    msg = f"Script `{path}` failed with following output:"
    if outs:
        msg = f"{msg}\nSTDOUT:\n{outs}"
    if errs:
        msg = f"{msg}\nSTDERR:\n{errs}"
    assert proc.returncode <= 0, msg
