"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os
import subprocess as sp
import sys
from pathlib import Path
from typing import Set

import pytest

from pde.tools.misc import module_available

PACKAGEPATH = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = PACKAGEPATH / "examples"
SKIP_EXAMPLES: Set[str] = set()
if not module_available("phasesep"):
    SKIP_EXAMPLES.add("droplets_active.py")
if not module_available("napari"):
    SKIP_EXAMPLES.add("droplets_interactive.py")
if not module_available("numba_scipy"):
    SKIP_EXAMPLES.add("droplets_active.py")


@pytest.mark.no_cover
@pytest.mark.skipif(sys.platform == "win32", reason="Assumes unix setup")
@pytest.mark.parametrize("path", EXAMPLE_PATH.glob("**/*.py"))
def test_example(path):
    """runs an example script given by path"""
    if path.name.startswith("_"):
        pytest.skip("skip examples starting with an underscore")
    if any(name in str(path) for name in SKIP_EXAMPLES):
        pytest.skip(f"Skip test {path}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGEPATH) + ":" + env.get("PYTHONPATH", "")
    proc = sp.Popen([sys.executable, path], env=env, stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        outs, errs = proc.communicate(timeout=30)
    except sp.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()

    msg = "Script `%s` failed with following output:" % path
    if outs:
        msg = "%s\nSTDOUT:\n%s" % (msg, outs)
    if errs:
        msg = "%s\nSTDERR:\n%s" % (msg, errs)
    assert proc.returncode <= 0, msg
