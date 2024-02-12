"""Stub to copy into a repl at ~ to import shorthand"""
import importlib.util as iu
import sys
_spec = iu.spec_from_file_location("shorthand", "code/py/shorthand/shorthand.py")
_mod = iu.module_from_spec(_spec)
sys.modules["shorthand"] = _mod
_spec.loader.exec_module(_mod)
from shorthand import *
