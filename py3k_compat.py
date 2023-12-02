import sys

if sys.version_info[0] >= 3:
  import collections.abc

  PY3 = True

  def callable(x):
    return isinstance(x, collections.abc.Callable)

  def execfile(fname, glob, loc=None):
    loc = loc if (loc is not None) else glob
    with open(fname) as fil:
      txt = fil.read()
    exec(compile(txt, fname, "exec"), glob, loc)

  unicode = str
  bytes = bytes
else:
  PY3 = False
  callable = callable
  execfile = execfile
  bytes = str
  unicode = unicode
