# Santa's Little Helpers
from functools import partial, reduce # just so they're on hand
from dataclasses import dataclass
from operator import itemgetter
from datetime import datetime
from itertools import chain
from random import sample
from pathlib import Path
from time import time
from math import prod # to have on hand (now pypy supports 3.8)
import sys

class dot(dict): # as in a "dot dict", a dict you can access by a "." # pretty inefficient bc. (dict), but convenient
  __getattr__, __setattr__ = dict.__getitem__, dict.__setitem__

def slots(self):
  return {slot:self.__getattribute__(slot) for slot in self.__slots__}

@dataclass
class data: # I recommend this pattern in general for POD, slightly more memory with __slots__ but guarantees perf
  __slots__ = ["x", "y", "z", "w"]
  x: float
  y: float
  z: float
  w: float
  def slots(self): # rather helpful for introspection or debugging
    return {slot:self.__getattribute__(slot) for slot in self.__slots__}
  def __dict__(self): # direct method isn't required, but faster, vars(d)() == d.slots()
    return {slot:self.__getattribute__(slot) for slot in self.__slots__}

try: # 3.10+
  @dataclass(slots=True)
  class quickdata: # Nicer, only some linters realise __slots__ get autogen'd
    x: float
    y: float
    z: float
    w: float
except:
  pass

def sorted_dict_by_key(d: dict): # sort a dict by key
  return dict(sorted(d.items(), key=itemgetter(0)))

def sorted_dict_by_val(d: dict): # sort a dict by value
  return dict(sorted(d.items(), key=itemgetter(1)))

def sortas(first: list, second: list): # sorts the first as if it was the second
  return list(map(itemgetter(0), sorted(zip(first,second), key=itemgetter(1))))

def bits(x: int): # because bin() has the annoying 0b, so slower but cleaner
  return f"{x:b}"

def ilog2(x): # integer log2, aka the position of the first bit
  return x.bit_length()-1

if sys.version_info.major >= 3 and sys.version_info.minor >= 10:
  def popcount(x: int):
    return x.bit_count() # yay
else:
  def popcount(x: int):
    return bin(x).count("1")

# these to/from bytes wrappers are just for dunder "ephemeral" bytes, use normal int.to/from when byteorder matters
def to_bytes(x: int, nbytes=None, signed=None, byteorder=sys.byteorder) -> bytes:
  "int.to_bytes but with (sensible) default values, assumes unsigned if positive, signed if negative"
  return x.to_bytes((nbytes or (x.bit_length()+7)//8), byteorder, signed=(abs(x)!=x) if signed is None else signed)

def from_bytes(b: bytes, signed=False, byteorder=sys.byteorder) -> int:
  "int.from_bytes but sensible byteorder, you must say if it's signed"
  return int.from_bytes(b, byteorder, signed)

def isqrt(n: int): # not as fast as int(sqrt) but works for all ints, not just those in f64 integer range
  x, y = n, (n + 1)//2
  while y < x:
    x, y = y, (y + n//y)//2
  return x

def isprime(n: int): # simple iterative one
  if n in {2, 3, 5, 7}:
    return True
  if not (n & 1) or not (n % 3) or not (n % 5) or not (n % 7):
    return False
  if n < 121:
    return n > 1
  sqrt = isqrt(n)
  assert(sqrt*sqrt <= n)
  for i in range(11, sqrt, 6):
    if not (n % i) or not (n % (i + 2)):
      return False
  return True

flatten = chain.from_iterable

def lmap(f, *args): # because wrapping in list() all the time is awkward, saves abusing the slow `*a,=map(*args)`!
  return list(map(f, *args))

def transpose(matrix): # not the fastest way, so if it's a big matrix please don't use
  return lmap(list, zip(*matrix))

def tmap(f, *args): # for the versions of python with faster tuple lookups (PEP 590 vectorcalls effect this how?)
  return tuple(map(f, *args))

def join(iterable, sep = " "): # because sep.join(it) doesn't convert str(it) for it in iterable
  return sep.join(map(str, iterable))

def avg(iterable, start=0): # because x/0 = 0 in euclidean
  return sum(iterable, start) / len(iterable) if len(iterable) else 0

def minmax(*iterable): # get the minimum and maximum quickly
  return min(iterable),max(iterable) # min(*iterable) is only faster for len(iterable) == 2

def minmax_ind(*iterable): # minmax but with indices, so ((i_a,min),(i_b,max))
  return min(enumerate(iterable),key=itemgetter(1)),max(enumerate(iterable),key=itemgetter(1))

def shuffled(*iterable): # aka, "shuffle but not in place", ie. reversed and sorted, but they ignored prior
  iterable = list(iterable) # this way we support sets
  return sample(iterable, len(iterable))

def lenfilter(iterable, pred=bool): # counts how many are true for a given predicate
  return sum(1 for it in iterable if pred(it)) # better (esp in pypy) than len(filter()) since not constructing a list

def first(iterable, default=None): # first item
  return next(iter(iterable), default) 

def dotprod(A, B):
  return sum(a*b for a,b in zip(A,B))

def sample_set(s: set, k: int): # 3.9 was doing so many things right, this is just annoying
  return sample(list(s), k) # if you really care about reproducibility (with known seeds) then sure, use sorted()

def now():
  return f"{datetime.now():%Y-%m-%d-%H-%M-%S}"

def tf(func, *args, **kwargs): # time func
  start = time()
  r = func(*args, **kwargs)
  end = time()
  print(f"{func.__qualname__}({', '.join(list(map(str,args)) + [f'{k}={v}' for k,v in kwargs.items()])}) = {r}, took {human_time(end-start)}")
  return r

def human_time(t: float, seconds = True): # because nobody makes it humanly readable
  return f"{int(t//60)}m {human_time((int(t)%60)+(t-int(t)), True)}" if t > 60 else \
         f"{t:.3f}s" if t > 0.1 and seconds else                                    \
         f"{t*1000:.3f}ms" if t > 0.0001 else                                       \
         f"{t*1000000:.3f}us"

def hours_minutes_seconds(t: float):
  seconds = int(t)
  print(f"{seconds}s")
  minutes,seconds = seconds//60, seconds%60
  print(f"{minutes}m{seconds}s")
  if minutes >= 60:
    hours, minutes = minutes//60, minutes%60
    print(f"{hours}h{minutes}m{seconds}s")
    if hours >= 24:
      days, hours = hours//24, hours%24
      print(f"{days}d{hours}h{minutes}m{seconds}s")
      if days >= 7:
        weeks, days = days//7, days%7
        print(f"{weeks}w{days}d{hours}h{minutes}m{seconds}s")
  print()

def find(v, iterable: list, start = 0, stop = -1, missing = -1): # find v in interable without exceptions
  try:
    return iterable.index(v, start, stop)
  except: # because if doesn't have .index then we couldn't find iterable
    return missing

def yesno(msg="", accept_return=True, replace_lists=False, yes_list=set(), no_list=set()):
  while True:
    reply = input(f"{msg} [y/N]: ").strip().lower()
    if reply in (yes_list if replace_lists else {"y", "ye", "yes"} | yes_list) or (accept_return and reply == ""): return True
    if reply in (no_list if replace_lists else {"n", "no"} | no_list): return False

def readlines(self: Path, hint=-1, encoding="utf8"): # just reads a line
  with self.open(encoding=encoding) as f:
    return f.readlines(hint)
