"""Keep the locked test suite's original imports pointing at package modules."""

import importlib
import sys


for module_name in (
    "layers", "window_cache", "encoder", "memory", "merge", "decoder",
    "model", "baselines", "task", "objectives",
):
    sys.modules[module_name] = importlib.import_module(f"rlt.{module_name}")
