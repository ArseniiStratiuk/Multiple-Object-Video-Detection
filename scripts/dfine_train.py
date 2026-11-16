"""Run D-FINE's own train.py with the visdet ops registered.

Usage (from the D-FINE checkout, under torchrun):
    VISDET_DFINE=$PWD torchrun --nproc_per_node=4 <repo>/scripts/dfine_train.py -c <config> ...
"""

import os
import runpy
import sys

sys.path.insert(0, os.environ["VISDET_DFINE"])

from visdet.models.dfine import register_ops  # noqa: E402

register_ops()
sys.argv[0] = os.path.join(os.environ["VISDET_DFINE"], "train.py")
runpy.run_path(sys.argv[0], run_name="__main__")
