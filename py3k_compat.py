import sys

if sys.version_info[0] >= 3:
  import collections.abc

  PY3 = True

  def callable(x):
    return isinstance(x, collections.abc.Callable)

  def execfile(fname, glob, loc=None):
    loc = loc if (loc is not None) else glob
    with open(fname, encoding="locale") as fil:  # noqa: FURB101
      txt = fil.read()
    exec(compile(txt, fname, "exec"), glob, loc)

  unicode = str
  bytes = bytes  # noqa: PLW0127
else:
  PY3 = False
  callable = callable  # noqa: PLW0127
  execfile = execfile  # noqa: PLW0127
  bytes = str
  unicode = unicode  # noqa: PLW0127
