'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''


import glob
import os
import sys
import subprocess as sp
from typing import List  # @UnusedImport
from pathlib import Path

import pytest
import numba as nb

from pde.tools.misc import module_available


PACKAGEPATH = Path(__file__).parents[2].resolve()
EXAMPLE_PATH = PACKAGEPATH / 'examples'



@pytest.mark.skipif(sys.platform == 'win32', reason="Assumes unix setup")
@pytest.mark.skipif(nb.config.DISABLE_JIT,
                    reason='pytest seems to check code coverage')
@pytest.mark.parametrize('path', glob.glob(str(EXAMPLE_PATH / '*.py')))
def test_examples(path):
    """ runs an example script given by path """
    if path.endswith('active_droplets.py') and not module_available('phasesep'):
        pytest.skip('The example active_droplets.py requires the `phasesep` '
                    'package')
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGEPATH) + ":" + env.get("PYTHONPATH", "")
    proc = sp.Popen([sys.executable, path], env=env, stdout=sp.PIPE,
                    stderr=sp.PIPE)
    try:
        outs, errs = proc.communicate(timeout=30)
    except sp.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()            
        
    msg = 'Script `%s` failed with following output:' % path
    if outs:
        msg = '%s\nSTDOUT:\n%s' % (msg, outs)
    if errs:
        msg = '%s\nSTDERR:\n%s' % (msg, errs)
    assert proc.returncode == 0, msg
