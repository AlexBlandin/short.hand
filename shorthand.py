"""
short.hand
---

Here we collect useful functions and classes, which are often "fast enough".
Star-importing this includes a few handy stdlib imports, great for the REPL.
"""

# Imports used here
from dataclasses import dataclass
from collections import ChainMap
from itertools import count, chain
from operator import itemgetter, attrgetter, indexOf
from datetime import datetime
from pathlib import Path
from random import randrange, sample
from time import time
import dataclasses
import hashlib

####################
# Import Shorthand #
####################
"""for when `from slh import *` is used"""

# ruff: noqa: E402 F401
from collections.abc import Sequence, Iterable
from functools import partial, reduce, cache
from typing import SupportsIndex, NamedTuple, Optional, Callable, Literal, Union, Any
from math import sqrt, prod
import itertools as it
import sys
import re
import os

RE_HTTP = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", flags = re.I | re.M | re.U) # credit @stephenhay

PY3 = sys.version_info.major >= 3
PY3_10_PLUS = PY3 and sys.version_info.minor >= 10
PY3_11_PLUS = PY3 and sys.version_info.minor >= 11

if PY3_10_PLUS:
  from itertools import pairwise # type: ignore
else:
  # we have to make it ourselves
  def pairwise(iterable: Iterable):
    """return an iterator of overlapping pairs taken from the input iterator `pairwise([1,2,3,4]) -> [(1,2), (2,3), (3,4)]`"""
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

#################
# POD Shorthand #
#################

class dot(dict):
  """a "dot dict", a dict you can access by a `.` - inefficient vs dataclass, but convenient"""
  __getattr__, __setattr__ = dict.__getitem__, dict.__setitem__ # type: ignore

@cache
def cls_to_tuple(cls):
  """this converts a class to a NamedTuple; cached because this is expensive!"""
  return NamedTuple(cls.__name__, **cls.__annotations__)

TRY_SLOTS_TRUE = {"slots": True} if PY3_10_PLUS else {}
"@dataclass(slots = True) only accessible in Python 3.10+"

# with these POD patterns, the provided methods for iteration and conversion to dicts/tuples are there to aid performance
# as dataclasses.astuple and dataclasses.asdict are an order of magnitude slower even with such a simple class, its deepcopy semantics versus our shallow copies
# this demonstrates a way to easily get great performance while reclaiming quality of life, see the performance testing at the end for an example of usage
# also, as noted since 3.11, we should not be using .__slots__ due to base class issues, but for performance, we do anyway (use dataclasses.fields, see method)

@dataclass(**TRY_SLOTS_TRUE)
class Struct:
  """a struct-like Plain Old Data base class, this is consistently much faster but breaks when subclassed, use StructSubclassable if you need that"""
  
  def __iter__(self):
    """iterating over the values, rather than the __slots__"""
    yield from map(self.__getattribute__, self.__slots__) # type: ignore
  
  def __len__(self):
    """how many slots there are, useful for slices, iteration, and reversing"""
    return len(self.__slots__) # type: ignore
  
  def __getitem__(self, n: Union[int, slice]):
    """generic __slots__[n] -> val, because subscripting (and slicing) is handy at times"""
    if isinstance(n, int):
      return self.__getattribute__(self.__slots__[n]) # type: ignore
    else:
      return list(map(self.__getattribute__, self.__slots__[n])) # type: ignore
  
  def _astuple(self):
    """generic __slots__ -> tuple; super fast, low quality of life"""
    return tuple(map(self.__getattribute__, self.__slots__)) # type: ignore
  
  def aslist(self):
    """generic __slots__ -> list; super fast, low quality of life, a shallow copy"""
    return list(map(self.__getattribute__, self.__slots__)) # type: ignore
  
  def asdict(self):
    """generic __slots__ -> dict; helpful for introspection, limited uses outside debugging"""
    return {slot: self.__getattribute__(slot) for slot in self.__slots__} # type: ignore
  
  def astuple(self):
    """generic __slots__ -> NamedTuple; a named shallow copy"""
    return cls_to_tuple(type(self))._make(map(self.__getattribute__, self.__slots__)) # type: ignore

@dataclass(**TRY_SLOTS_TRUE)
class StructSubclassable:
  """a struct-like Plain Old Data base class, we recommend this approach, this has consistently "good" performance and can still be subclassed"""
  
  def __iter__(self):
    """iterating over the values, rather than the __slots__"""
    yield from map(self.__getattribute__, self.fields())
  
  def __len__(self):
    """how many slots there are, useful for slices, iteration, and reversing"""
    return len(self.fields())
  
  def __getitem__(self, n: Union[int, slice]):
    """generic __slots__[n] -> val, because subscripting (and slicing) is handy at times"""
    if isinstance(n, int):
      return self.__getattribute__(self.fields()[n])
    else:
      return list(map(self.__getattribute__, self.fields()[n]))
  
  def _astuple(self):
    """generic __slots__ -> tuple; super fast, low quality of life, a shallow copy"""
    return tuple(map(self.__getattribute__, self.fields()))
  
  def aslist(self):
    """generic __slots__ -> list; super fast, low quality of life, a shallow copy"""
    return list(map(self.__getattribute__, self.fields()))
  
  def asdict(self):
    """generic __slots__ -> dict; helpful for introspection, limited uses outside debugging, a shallow copy"""
    return {slot: self.__getattribute__(slot) for slot in self.fields()}
  
  def astuple(self):
    """generic __slots__ -> NamedTuple; nicer but just slightly slower than asdict"""
    return cls_to_tuple(type(self))._make(map(self.__getattribute__, self.fields()))
  
  def fields(self):
    """__slots__ equivalent using the proper fields approach"""
    return list(map(attrgetter("name"), dataclasses.fields(self)))

#######################
# Iterables Shorthand #
#######################

flatten = chain.from_iterable

class Circular(list):
  """a circularly addressable list, where Circular([0, 1, 2, 3, 4])[-5:10] is [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]"""
  def __getitem__(self, x: Union[int, slice]):
    if isinstance(x, slice):
      return [self[x] for x in range(0 if x.start is None else x.start, len(self) if x.stop is None else x.stop, 1 if x.step is None else x.step)]
    return super().__getitem__(x.__index__() % max(1, len(self)))
  
  def __setitem__(self, x: Union[SupportsIndex, slice], val):
    if isinstance(x, slice) and (hasattr(val, "__iter__") or hasattr(val, "__getitem__")):
      m = max(1, len(self))
      for i, v in zip(count(0 if x.start is None else x.start, 1 if x.step is None else x.step), val) if x.stop is None else zip(
        range(0 if x.start is None else x.start,
              len(self) if x.stop is None else x.stop, 1 if x.step is None else x.step), val
      ):
        super().__setitem__(i.__index__() % m, v)
    else:
      super().__setitem__(x.__index__() % max(1, len(self)), val) # type: ignore
  
  def repeat(self, times: Optional[int] = None):
    if times is None:
      while True:
        yield from iter(self)
    else:
      for _ in range(times):
        yield from iter(self)

def unique_list(xs: Sequence):
  """reduce a list to only its unique elements `[1,1,2,7,2,4] -> [1,2,7,4]`; can be passed as vargs or a single list, for convenience"""
  return list(dict(zip(xs if len(xs) != 1 else xs[0], it.count())))

def unwrap(f: Callable, *args, **kwargs):
  """because exceptions are bad"""
  try:
    return f(*args, **kwargs)
  except Exception:
    return None

def compose(*fs: Callable):
  """combine each function in fs; evaluates fs[0] first, and fs[-1] last, like fs[-1](fs[-2](...fs[0](*args, **kwargs)...))"""
  def _comp(x):
    # for the sake of simplicity, it assumes an arity of 1 for every function, because it might want a tuple in, or vargs, who knows
    for f in fs:
      x = f(x)
    return x
  
  return _comp

def mapcomp(xs: Iterable, *fs: Callable):
  """map(compose(*fs), iterable); evaluates fs[0] first, fs[-1] last, so acts like map(fs[-1], map(fs[-2], ... map(fs[0], iterable)...))"""
  def _comp(fs: list):
    # not using compose() internally to avoid overhead, this is faster than list(map(compose(*fs), iterable))
    if len(fs):
      f = fs.pop()
      return map(f, _comp(fs))
    return xs
  
  return list(_comp(list(fs)))

def lmap(f: Callable, *args):
  """because wrapping in list() all the time is awkward, saves abusing the slow `*a,=map(*args)`!"""
  return list(map(f, *args))

def transpose(matrix: list[list]):
  """inefficient but elegant, so if it's a big matrix please don't use"""
  return lmap(list, zip(*matrix))

def tmap(f: Callable, *args):
  """for the versions of python with faster tuple lookups"""
  return tuple(map(f, *args))

def join(xs: Iterable, sep = " "):
  """because sep.join(iterable) doesn't convert to str(i) for i in iterable"""
  return sep.join(map(str, xs))

def minmax(xs: Iterable):
  """get the minimum and maximum quickly"""
  return min(xs), max(xs)

def minmax_ind(xs: Iterable):
  """minmax but with indices, so ((i_a, min), (i_b, max))"""
  return min(enumerate(xs), key = itemgetter(1)), max(enumerate(xs), key = itemgetter(1))

def shuffled(xs: Iterable):
  """aka, "shuffle but not in place", like reversed() and sorted()"""
  xs = list(xs) # this way we support sets, without a sort, as sample doesn't anymore
  return sample(xs, len(xs))

def lenfilter(xs: Iterable, pred = bool):
  """counts how many are true for a given predicate"""
  return sum(1 for i in xs if pred(i)) # better (esp in pypy) than len(filter()) since not constructing a list

def first(xs: Iterable, default = None):
  """the first item in iterable"""
  return next(iter(xs), default)

def sample_set(s: set, k: int):
  """sample a set because you just want some random elements and don't care (about reproducibility)"""
  return sample(list(s), k) # if you care about reproducibility (with known seeds), sort prior

def sorted_dict_by_key(d: dict, reverse = False):
  """sort a dict by key"""
  return dict(sorted(d.items(), key = itemgetter(0), reverse = reverse))

def sorted_dict_by_val(d: dict, reverse = False):
  """sort a dict by value"""
  return dict(sorted(d.items(), key = itemgetter(1), reverse = reverse))

def sorted_dict(d: dict, key = itemgetter(1), reverse = False):
  """generic sorting, because it's something people kinda want"""
  return dict(sorted(d.items(), key = key, reverse = reverse))

def sortas(first: Iterable, second: Iterable):
  """sorts the first as if it was the second"""
  return list(map(itemgetter(0), sorted(zip(first, second))))

def dedupe(it):
  """deduplicates an iterator, consumes memory to do so, non-blocking"""
  s = set()
  for i in it:
    if i not in s:
      s.add(i)
      yield i

def find(v, xs: Union[list, Iterable], start: Optional[int] = None, stop: Optional[int] = None, missing = -1):
  """find the first index of v in interable without raising exceptions, will consume iterables so be careful"""
  if isinstance(xs, list):
    return xs.index(v, start if start is not None else 0, stop if stop is not None else sys.maxsize)
  else:
    try:
      if (start is None or start == 0) and (stop is None or stop == -1):
        return indexOf(xs, v)
      else:
        return list(xs).index(v, start if start is not None else 0, stop if stop is not None else sys.maxsize)
    except Exception:
      return missing

class DeepChainMap(ChainMap):
  """Variant of ChainMap that allows direct updates to inner scopes"""
  def __setitem__(self, key, value):
    for mapping in self.maps:
      if key in mapping:
        mapping[key] = value
        return # we only modify the first match
    self.maps[0][key] = value
  
  def __delitem__(self, key):
    for mapping in self.maps:
      if key in mapping:
        del mapping[key]
        return # we only modify the first match
    raise KeyError(key)

###################
# Maths Shorthand #
###################

def avg(xs: Sequence, start = 0):
  """no exceptions, because x/0 = 0 in euclidean"""
  return sum(xs, start) / len(xs) if len(xs) else 0

def dotprod(A: Iterable, B: Iterable):
  return sum(a * b for a, b in zip(A, B))

def bits(x: int):
  """because bin() has the annoying 0b, so slower but cleaner"""
  return f"{x:b}"

def ilog2(x: int):
  """integer log2, aka the position of the first bit"""
  return x.bit_length() - 1

# from gmpy2 import bit_scan1 as ctz # if you must go faster
def ctz(v: int):
  """count trailing zeroes"""
  return (v & -v).bit_length() - 1

if PY3_10_PLUS:
  # we get the fast builtin!
  def popcount(x: int):
    """Number of ones in the binary representation of the absolute value of self."""
    return x.bit_count()
else:
  # we use the slow string version
  def popcount(x: int):
    """Number of ones in the binary representation of the absolute value of self."""
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

def fastprime(n: int, trials: int = 8):
  """
  Miller-Rabin primality test.

  - Returns False when n is not prime.
  - Returns True when n is a prime under 3317044064679887385961981, else when n is very likely a prime.
  
  Increase the number of trials to increase the confidence for n >= 3317044064679887385961981 at cost to performance
  """
  
  if n in {2, 3, 5, 7}:
    return True
  if not (n & 1) or not (n % 3) or not (n % 5) or not (n % 7):
    return False
  if n < 121:
    return n > 1
  
  d = n - 1
  s = ctz(d)
  d >>= s
  
  # assert(2**s * d == n-1) # not necessary, but go for it if you want
  
  def witness(a):
    if pow(a, d, n) == 1:
      return False
    for i in range(s):
      if pow(a, 2**i * d, n) == n - 1:
        return False
    return True
  
  if n < 2047:
    b = [2]
  elif n < 1373653:
    b = [2, 3]
  elif n < 9080191:
    b = [31, 73]
  elif n < 25326001:
    b = [2, 3, 5]
  elif n < 3215031751:
    b = [2, 3, 5, 7]
  elif n < 4759123141:
    b = [2, 7, 61]
  elif n < 1122004669633:
    b = [2, 13, 23, 1662803]
  elif n < 2152302898747:
    b = [2, 3, 5, 7, 11]
  elif n < 3474749660383:
    b = [2, 3, 5, 7, 11, 13]
  elif n < 341550071728321:
    b = [2, 3, 5, 7, 11, 13, 17]
  elif n < 318665857834031151167461:
    b = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] # covers 64bit
  elif n < 3317044064679887385961981:
    b = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
  else:
    b = [2] + [randrange(3, n, 2) for _ in range(trials)]
  
  for a in b:
    if witness(a):
      return False
  return True

####################
# Timing Shorthand #
####################

def now():
  """because sometimes I want the time now()"""
  return f"{datetime.now():%Y-%m-%d-%H-%M-%S}"

def tf(func: Callable, *args, __pretty_tf = True, **kwargs):
  """time func func, as in, func to time the function func"""
  start = time()
  r = func(*args, **kwargs)
  end = time()
  if __pretty_tf:
    fargs = list(map(str, map(lambda a: a.__name__ if hasattr(a, '__name__') else a, args))) + [f'{k}={v}' for k, v in kwargs.items()]
    print(f"{func.__qualname__}({', '.join(fargs)}) = {r} ({human_time(end-start)})")
  else:
    print(human_time(end - start))
  return r

def human_time(t: float, seconds = True):
  """because nobody makes it humanly readable"""
  return f"{t//60:.0f}m {t%60:.3f}s" if t > 60 else f"{t:.3f}s" if t > 0.1 and seconds else f"{t*1000:.3f}ms" if t > 0.0001 else f"{t*10**6:.3f}us"

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
    if reply in (no_list if replace_lists else {"n", "no"} | no_list):
      return False

# these to/from bytes wrappers are just for dunder "ephemeral" bytes, use normal int.to/from when byteorder matters
def to_bytes(x: int, nbytes: Optional[int] = None, signed: Optional[bool] = None, byteorder: Literal["little", "big"] = sys.byteorder) -> bytes:
  """int.to_bytes but with (sensible) default values, by default assumes unsigned if >=0, signed if <0"""
  return x.to_bytes((nbytes or (x.bit_length() + 7) // 8), byteorder, signed = (x >= 0) if signed is None else signed)

def from_bytes(b: bytes, signed: bool = False, byteorder: Literal["little", "big"] = sys.byteorder) -> int:
  """int.from_bytes but sensible byteorder, you must say if it's signed"""
  return int.from_bytes(b, byteorder, signed = signed)

if PY3_11_PLUS:
  file_digest = hashlib.file_digest # type: ignore
else:
  def file_digest(fileobj, digest, /, *, _bufsize = 2**18):
    """Hash the contents of a file-like object. Returns a digest object. Backport from 3.11."""
    if isinstance(digest, str):
      digestobj = hashlib.new(digest)
    else:
      digestobj = digest()
    
    if hasattr(fileobj, "getbuffer"):
      digestobj.update(fileobj.getbuffer())
      return digestobj
    
    if not (hasattr(fileobj, "readinto") and hasattr(fileobj, "readable") and fileobj.readable()):
      raise ValueError(f"'{fileobj!r}' is not a file-like object in binary reading mode.")
    
    buf = bytearray(_bufsize)
    view = memoryview(buf)
    while True:
      size = fileobj.readinto(buf)
      if size == 0:
        break # EOF
      digestobj.update(view[:size])
    
    return digestobj

##################
# Path Shorthand #
##################

# convenience functions to not write as much

def resolve(path: Union[str, Path]):
  """resolve a Path including "~" (bc Path(path) doesn't...)"""
  return Path(path).expanduser()

@cache
def filedigest(path: Path, hash = "sha1"):
  """fingerprint a file, caches so modified files bypass with filedigest.__wrapped__ or filedigest.cache_clear()"""
  with open(path, "rb") as f:
    return file_digest(f, hash).hexdigest()

def readlines(fp: Union[str, Path], encoding = "utf8"):
  """just reads lines as you normally would want to"""
  return resolve(fp).read_text(encoding).splitlines()

def readlinesmap(fp: Union[str, Path], *fs: Callable, encoding = "utf8"):
  """readlines but map each function in fs to fp's lines in order (fs[0]: first, ..., fs[-1]: last)"""
  return mapcomp(resolve(fp).read_text(encoding).splitlines(), *fs)

def writelines(fp: Union[str, Path], lines: Union[str, list[str]], encoding = "utf8", newline = "\n"):
  """just writes lines as you normally would want to"""
  return resolve(fp).write_text(lines if isinstance(lines, str) else newline.join(lines), encoding = encoding, newline = newline)

def writelinesmap(fp: Union[str, Path], lines: Union[str, list[str]], *fs: Callable, encoding = "utf8", newline = "\n"):
  """writelines but map each function in fs to fp's lines in order (fs[0] first, fs[-1] last)"""
  return (resolve(fp).write_text(newline.join(mapcomp(lines if isinstance(lines, list) else lines.splitlines()), *fs), encoding = encoding, newline = newline))

####################
# String Shorthand #
####################

def lev(s1: str, s2: str) -> int:
  """calculate Levenshtein distance between strings"""
  if s1 == s2:
    return 0
  l1, l2 = len(s1), len(s2)
  if 0 in (l1, l2):
    return l1 or l2
  if l1 > l2:
    s1, s2, l1, l2 = s2, s1, l2, l1
  d0, d1 = list(range(l2 + 1)), list(range(l2 + 1))
  for i, x in enumerate(s1):
    d1[0] = i + 1
    for j, y in enumerate(s2):
      cost = d0[j]
      if x != y:
        cost += 1
        ins_cost = d1[j] + 1
        del_cost = d0[j + 1] + 1
        if ins_cost < cost:
          cost = ins_cost
        if del_cost < cost:
          cost = del_cost
      d1[j + 1] = cost
    d0, d1 = d1, d0
  return d0[-1]

#########################
# Performance & Testing #
#########################

if __name__ == "__main__":
  from dataclasses import astuple, asdict # noqa: F401
  from timeit import repeat
  
  N_RUNS, N_ITERATIONS = 10, 10**6
  
  for Base in Struct, StructSubclassable:
    if PY3_10_PLUS:
      
      @dataclass(slots = True)
      class Vec4(Base): # type: ignore
        x: float
        y: float
        z: float
        w: float
    else: # you must manually specify __slots__ as @dataclass(slots = True) was only added in 3.10
      
      @dataclass
      class Vec4(Base): # type: ignore
        __slots__ = ["x", "y", "z", "w"]
        x: float
        y: float
        z: float
        w: float
    
    d = Vec4(1.2, 3.4, 5.6, 7.8) # type: ignore  # noqa: F405
    tests = [
      "tuple(d)    ", # FastStruct: py 0.527s pypy 0.119s, Struct: py 2.379s pypy 0.469s # tuple around .__iter__
      "astuple(d)  ", # FastStruct: py 4.539s pypy 0.631s, Struct: py 4.604s pypy 0.632s # dataclasses.astuple
      "d._astuple()", # FastStruct: py 0.340s pypy 0.106s, Struct: py 1.310s pypy 0.407s # shallow copy version of dataclasses.astuple
      "d.astuple() ", # FastStruct: py 0.602s pypy 0.248s, Struct: py 1.651s pypy 0.582s # d._astuple() but returns a namedtuple of d
      "asdict(d)   ", # FastStruct: py 4.838s pypy 0.775s, Struct: py 5.404s pypy 0.784s # dataclasses.asdict
      "d.asdict()  ", # FastStruct: py 0.519s pypy 0.175s, Struct: py 1.565s pypy 0.463s # shallow copy version of dataclasses.asdict
      "d[0]        ", # FastStruct: py 0.147s pypy 0.010s, Struct: py 1.160s pypy 0.285s # typical operator
      "d[-1]       ", # FastStruct: py 0.147s pypy 0.011s, Struct: py 1.168s pypy 0.286s # typical operator
      "d[:]        ", # FastStruct: py 0.423s pypy 0.098s, Struct: py 1.400s pypy 0.396s # typical operator
      "list(d)     ", # FastStruct: py 0.558s pypy 0.106s, Struct: py 3.300s pypy 0.715s # much slower than the [:] operator
      "d.aslist()  ", # FastStruct: py 0.350s pypy 0.082s, Struct: py 1.427s pypy 0.409s # comparable to the [:] operator
      "d[::-1]     ", # FastStruct: py 0.454s pypy 0.107s, Struct: py 1.403s pypy 0.403s # typical operator
      "d[:1]       ", # FastStruct: py 0.342s pypy 0.043s, Struct: py 1.332s pypy 0.330s # typical operator
    ]
    
    print(f"{Base.__name__} ({N_RUNS} runs):")
    for code in tests: # actual run
      print(f"  {code} # {min(repeat(code, number = N_ITERATIONS, repeat = N_RUNS, globals = globals())):0.3f}s")
