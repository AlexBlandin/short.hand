"""
Stub to copy into a repl at ~ to import shorthand.

Copyright 2024 Alex Blandin
"""
# ruff: noqa
# fmt: off

import importlib.util as iu
import sys

_spec = iu.spec_from_file_location("shorthand", "code/py/shorthand/shorthand.py")
_mod = iu.module_from_spec(_spec) # type: ignore
sys.modules["shorthand"] = _mod
_spec.loader.exec_module(_mod) # type: ignore
from shorthand import *
