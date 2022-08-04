"""
Santa's Little Helpers
"""

# Imports used here
from operator import indexOf, itemgetter
from random import sample, randrange
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path
from time import time

# Typing imports
from collections.abc import Iterable

####################
# Import Shorthand #
####################

"""for when `from slh import *` is used"""
from functools import partial, reduce, cache # just so they're on hand
from math import prod, sqrt # good to have on hand
import itertools as it # for it.count() etc
import sys, os

PY3_10 = sys.version_info.major >= 3 and sys.version_info.minor >= 10

if PY3_10:
  from itertools import pairwise
else:
  def pairwise(iterable: Iterable):
    """return an iterator of overlapping pairs taken from the input iterator `pairwise([1,2,3,4]) -> [(1,2), (2,3), (3,4)]`"""
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

#################
# POD Shorthand #
#################

class dot(dict):
  """a "dot dict", a dict you can access by a "." # pretty inefficient bc. (dict), but convenient"""
  __getattr__, __setattr__ = dict.__getitem__, dict.__setitem__

@dataclass
class data:
  """I recommend this pattern in general for POD, slightly more memory with __slots__ but guarantees perf"""
  __slots__ = ["x", "y", "z", "w"]
  x: float
  y: float
  z: float
  w: float
  
  def slots(self):
    """reconstruct the __slots__ dict of a class; rather helpful for introspection or debugging"""
    return {slot: self.__getattribute__(slot) for slot in self.__slots__}
  
  def __dict__(self):
    """direct method isn't required, but faster, vars(d)() == d.slots()"""
    return {slot: self.__getattribute__(slot) for slot in self.__slots__}

if PY3_10: # 3.10+
  @dataclass(slots = True)
  class quickdata:
    """cleaner, but only some linters realise __slots__ get autogen'd right now"""
    x: float
    y: float
    z: float
    w: float
else:
  pass

#######################
# Iterables Shorthand #
#######################

flatten = chain.from_iterable

def unique_list(*lst): 
  """reduce a list to only its unique elements `[1,1,2,7,2,4] -> [1,2,7,4]`; can be passed as vargs or a single list, for convenience"""
  return list(dict(lst if len(lst)!=1 else lst[0], it.count()))

def compose(*fs):
  """combine each function in fs; evaluates fs[0] first, and fs[-1] last, like fs[-1](fs[-2](...fs[0](*args, **kwargs)...))"""
  def comp(x):
    # for the sake of simplicity, it assumes an arity of 1 for every function, because it might want a tuple in, or vargs, who knows
    for f in fs:
      x = f(x)
    return x
  
  return comp

def mapcomp(iterable, *fs):
  """map(compose(*fs), iterable); evaluates fs[0] first, fs[-1] last, so acts like map(fs[-1], map(fs[-2], ... map(fs[0], iterable)...))"""
  def comp(fs: list):
    # not using compose() internally to avoid overhead, this is faster than list(map(compose(*fs), iterable))
    if len(fs):
      f = fs.pop()
      return map(f, comp(fs))
    return iterable
  
  return list(comp(list(fs)))

def lmap(f, *args):
  """because wrapping in list() all the time is awkward, saves abusing the slow `*a,=map(*args)`!"""
  return list(map(f, *args))

def transpose(matrix: list[list]):
  """inefficient but elegant, so if it's a big matrix please don't use"""
  return lmap(list, zip(*matrix))

def tmap(f, *args):
  """for the versions of python with faster tuple lookups (TODO: PEP 590 vectorcalls affect this how?)"""
  return tuple(map(f, *args))

def join(iterable, sep = " "):
  """because sep.join(iterable) doesn't convert to str(i) for i in iterable"""
  return sep.join(map(str, iterable))

def minmax(*iterable):
  """get the minimum and maximum quickly"""
  return min(iterable), max(iterable) # min(*iterable) is only faster for len(iterable) == 2

def minmax_ind(*iterable):
  """minmax but with indices, so ((i_a,min),(i_b,max))"""
  return min(enumerate(iterable), key = itemgetter(1)), max(enumerate(iterable), key = itemgetter(1))

def shuffled(*iterable):
  """aka, "shuffle but not in place", ie. reversed and sorted, but they ignored prior"""
  iterable = list(iterable) # this way we support sets
  return sample(iterable, len(iterable))

def lenfilter(iterable, pred = bool):
  """counts how many are true for a given predicate"""
  return sum(1 for i in iterable if pred(i)) # better (esp in pypy) than len(filter()) since not constructing a list

def first(iterable, default = None):
  """the first item in iterable"""
  return next(iter(iterable), default)

def sample_set(s: set, k: int):
  """sample a set because you just want some random elements and don't care (about reproducibility)"""
  return sample(list(s), k) # if you really care about reproducibility (with known seeds) then sure, use sorted()

def sorted_dict_by_key(d: dict, reverse = False):
  """sort a dict by key"""
  return dict(sorted(d.items(), key = itemgetter(0), reverse = reverse))

def sorted_dict_by_val(d: dict, reverse = False):
  """sort a dict by value"""
  return dict(sorted(d.items(), key = itemgetter(1), reverse = reverse))

def sorted_dict(d, key = itemgetter(1), reverse = False):
  """generic sorting, because it's something people kinda want"""
  return dict(sorted(d.items(), key = key, reverse = reverse))

def sortas(first: list, second: list):
  """sorts the first as if it was the second"""
  return list(map(itemgetter(0), sorted(zip(first, second), key = itemgetter(1))))

def find(v, iterable: list, start = 0, stop = -1, missing = -1):
  """find the first index of v in interable without raising exceptions"""
  try:
    return iterable.index(v, start, stop)
  except: # because if doesn't have .index then we couldn't find iterable
    if start == 0 and stop == -1: # unless we aren't messing with start and stop, we might have a chance
      try:
        indexOf(iterable, v)
      except:
        pass
    return missing

###################
# Maths Shorthand #
###################

def avg(iterable, start = 0):
  """without exceptions, because x/0 = 0 in euclidean"""
  return sum(iterable, start) / len(iterable) if len(iterable) else 0

def dotprod(A, B):
  return sum(a * b for a, b in zip(A, B))

def bits(x: int):
  """because bin() has the annoying 0b, so slower but cleaner"""
  return f"{x:b}"

def ilog2(x):
  """integer log2, aka the position of the first bit"""
  return x.bit_length() - 1

# from gmpy2 import bit_scan1 as ctz # if you must go faster
def ctz(v):
  """count trailing zeroes"""
  return (v & -v).bit_length() - 1

if PY3_10:
  def popcount(x: int):
    return x.bit_count() # yay
else:
  def popcount(x: int):
    return bin(x).count("1")

def isqrt(n: int):
  """works for all ints, fast for numbers < 2**52 (aka, abusing double precision sqrt)"""
  if n < 2**52:
    return int(sqrt(n))
  n = int(n)
  x, y = n, (n + 1) // 2
  while y < x:
    x, y = y, (y + n // y) // 2
  return x

def isprime(n: int):
  """simple iterative one"""
  if n in {2, 3, 5, 7}:
    return True
  if not (n & 1) or not (n % 3) or not (n % 5) or not (n % 7):
    return False
  if n < 121:
    return n > 1
  sqrt = isqrt(n)
  assert (sqrt * sqrt <= n)
  for i in range(11, sqrt, 6):
    if not (n % i) or not (n % (i + 2)):
      return False
  return True

def fastprime(n: int, trials = 8):
  """
  Miller-Rabin primality test.

  - Returns False when n is not prime.
  - Returns True when n is a prime under 3317044064679887385961981, else when n is very likely a prime.
  
  Increase the number of trials to increase the confidence for n >= 3317044064679887385961981 at cost to performance
  """
  
  if n in {2, 3, 5, 7}: return True
  if not (n & 1) or not (n % 3) or not (n % 5) or not (n % 7): return False
  if n < 121: return n > 1
  
  d = n - 1
  s = ctz(d)
  d >>= s
  
  # assert(2**s * d == n-1) # not necessary, but go for it if you want
  
  def witness(a):
    if pow(a, d, n) == 1: return False
    for i in range(s):
      if pow(a, 2**i * d, n) == n - 1: return False
    return True
  
  if n < 2047: b = [2]
  elif n < 1373653: b = [2, 3]
  elif n < 9080191: b = [31, 73]
  elif n < 25326001: b = [2, 3, 5]
  elif n < 3215031751: b = [2, 3, 5, 7]
  elif n < 4759123141: b = [2, 7, 61]
  elif n < 1122004669633: b = [2, 13, 23, 1662803]
  elif n < 2152302898747: b = [2, 3, 5, 7, 11]
  elif n < 3474749660383: b = [2, 3, 5, 7, 11, 13]
  elif n < 341550071728321: b = [2, 3, 5, 7, 11, 13, 17]
  elif n < 318665857834031151167461: b = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] # covers 64bit
  elif n < 3317044064679887385961981: b = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
  else: b = [2] + [randrange(3, n, 2) for _ in range(trials)]
  
  for a in b:
    if witness(a): return False
  return True

####################
# Timing Shorthand #
####################

def now():
  """because sometimes I want the time now()"""
  return f"{datetime.now():%Y-%m-%d-%H-%M-%S}"

def tf(func, *args, __pretty_tf = True, **kwargs):
  """time func func, as in, time the function func"""
  start = time()
  r = func(*args, **kwargs)
  end = time()
  if __pretty_tf:
    print(
      f"{func.__qualname__}({', '.join(list(map(str,args)) + [f'{k}={v}' for k,v in kwargs.items()])}) = {r}, took {human_time(end-start)}"
    )
  else:
    print(human_time(end - start))
  return r

def human_time(t: float, seconds = True):
  """because nobody makes it humanly readable"""
  return f"{int(t//60)}m {human_time((int(t)%60)+(t-int(t)), True)}" if t > 60 else \
         f"{t:.3f}s" if t > 0.1 and seconds else                                    \
         f"{t*1000:.3f}ms" if t > 0.0001 else                                       \
         f"{t*1000000:.3f}us"

def hours_minutes_seconds(t: float):
  """from some number of seconds t, how many (years) (weeks) (days) (hours) minutes and seconds are there (filled in as needed)"""
  seconds = int(t)
  print(f"{seconds}s")
  minutes, seconds = seconds // 60, seconds % 60
  print(f"{minutes}m{seconds}s")
  if minutes >= 60:
    hours, minutes = minutes // 60, minutes % 60
    print(f"{hours}h{minutes}m{seconds}s")
    if hours >= 24:
      days, hours = hours // 24, hours % 24
      print(f"{days}d{hours}h{minutes}m{seconds}s")
      if days >= 7:
        weeks, days = days // 7, days % 7
        print(f"{weeks}w{days}d{hours}h{minutes}m{seconds}s")
        if weeks >= 52:
          years, weeks = weeks // 52, weeks % 52
          print(f"{years}y{weeks}w{days}d{hours}h{minutes}m{seconds}s")
  print()

################
# IO Shorthand #
################

def yesno(msg = "", accept_return = True, replace_lists = False, yes_list = set(), no_list = set()):
  """keep asking until they say yes or no"""
  while True:
    reply = input(f"{msg} [y/N]: ").strip().lower()
    if reply in (yes_list if replace_lists else {"y", "ye", "yes"} | yes_list) or (accept_return and reply == ""):
      return True
    if reply in (no_list if replace_lists else {"n", "no"} | no_list): return False

# these to/from bytes wrappers are just for dunder "ephemeral" bytes, use normal int.to/from when byteorder matters
def to_bytes(x: int, nbytes = None, signed = None, byteorder = sys.byteorder) -> bytes:
  """int.to_bytes but with (sensible) default values, by default assumes unsigned if >=0, signed if <0"""
  return x.to_bytes((nbytes or (x.bit_length() + 7) // 8),
                    byteorder,
                    signed = (x >= 0) if signed is None else signed)

def from_bytes(b: bytes, signed = False, byteorder = sys.byteorder) -> int:
  """int.from_bytes but sensible byteorder, you must say if it's signed"""
  return int.from_bytes(b, byteorder, signed)

##################
# Path Shorthand #
##################

# convenience functions to not write as much

def resolve(path: str | Path):
  """resolve Path including "~" (bc Path(path) doesn't...)"""
  return Path(os.path.expanduser(path))

def readlines(fp: str | Path, encoding = "utf8"):
  """just reads lines as you normally would want to"""
  return Path(fp).read_text(encoding).splitlines()

def readlinesmap(fp: str | Path, *fs, encoding = "utf8"):
  """readlines but map each function in fs to fp's lines in order (fs[0]: first, ..., fs[-1]: last)"""
  return mapcomp(Path(fp).read_text(encoding).splitlines(), *fs)

def writelines(fp: str | Path, lines: str | list[str], encoding = "utf8", newline="\n"):
  """just writes lines as you normally would want to"""
  return Path(fp).write_text(lines if isinstance(lines, str) else newline.join(lines), encoding=encoding, newline=newline)

def writelinesmap(fp: str | Path, lines: str | list[str], *fs, encoding = "utf8", newline="\n"):
  """writelines but map each function in fs to fp's lines in order (fs[0] first, fs[-1] last)"""
  return (Path(fp).write_text(newline.join(mapcomp(lines if isinstance(lines, list) else lines.splitlines()), *fs), encoding=encoding, newline=newline))
