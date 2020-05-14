"""Set things up for building terra mystica models
Right now it just sets the path where d6tflow will save results
"""
from pathlib import Path

import d6tflow

_d6tflow_dir = Path(__file__).resolve().parents[1] / "d6tflow_output"
d6tflow.set_dir(str(_d6tflow_dir))
