"""
Fixes the compat from the defunct https://pypi.org/project/pyreadline/.

The only thing that needs setting is unicode as str, and the horrific execfile.
All else was redundant, slower, or actually worse than the builtin callable.
Oh, and Python 2 is dead and buried. So that simplified things.

Copyright 2020 Alex Blandin
"""

PY3 = True


def execfile(fname, glob, loc=None):
  loc = loc if (loc is not None) else glob
  with open(fname, encoding="locale") as fil:
    txt = fil.read()
  exec(compile(txt, fname, "exec"), glob, loc)


unicode = str
