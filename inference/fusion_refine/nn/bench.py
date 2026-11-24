"""benchmark/run_benchmark.py with this sub-project's methods registered (nn_methods.py) and the dataset /
highlighting paths of the hair_work workspace. Same arguments as run_benchmark.py."""
import sys

import fusion_methods  # noqa: F401  (benchmark methods)
try:
    import nn_methods  # noqa: F401  (registers nn_* methods)
except ImportError as e:   # before the model exists
    print("nn_methods not loaded:", e)
import os
import run_benchmark

# the guarded child processes must also come through this file (else the nn_* methods are unknown there)
_orig_argv = run_benchmark.worker_argv


def _worker_argv(*a, **k):
    argv = _orig_argv(*a, **k)
    argv[1] = os.path.abspath(__file__)
    return argv


run_benchmark.worker_argv = _worker_argv
run_benchmark.main()
