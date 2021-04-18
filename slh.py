# Santa's Little Helpers
from dataclasses import dataclass
from functools import partial
from itertools import chain
from random import sample
from math import log2
import sys

if sys.version_info.major >= 3 and sys.version_info.minor >= 8:
  from math import prod
else:
  from functools import reduce
  from operator import mul
  def prod(iterable, *, start=1): return reduce(mul, chain([start] if len(iterable)!=1 else [], iterable)) # not 3.8 (ie, pypy)

class dot(dict): # as in a "dot dict", a dict you can access by a "." # pretty inefficient bc. (dict), but elegant
  __getattr__, __setattr__ = dict.__getitem__, dict.__setitem__

@dataclass
class data: # I recommend this pattern in general for POD, slightly more memory with __slots__ but guarantees perf
  __slots__ = ["x", "y", "z", "w"]
  x: float
  y: float
  z: float
  w: float
  def slots(self): # Rather helpful for introspection or debugging
    return {slot:self.__getattribute__(slot) for slot in self.__slots__}

# these to/from bytes wrappers are just for dunder "ephemeral" bytes, use normal int.to/from when byteorder matters
def to_bytes(x: int, nbytes=None, byteorder=sys.byteorder, signed=None) -> bytes: # int.to_bytes but with (sensible) default values, assumes unsigned if positive, signed if negative
  return x.to_bytes(((x.bit_length()+7)//8) if nbytes is None else nbytes, byteorder, signed=(abs(x)!=x) if signed is None else signed)

def from_bytes(b: bytes, byteorder=sys.byteorder, signed=False) -> int: # you need to say if it's signed or not
  return int.from_bytes(b, byteorder, signed)

def isqrt(n): # not as fast as int(sqrt) but works for all ints, not just those in f64 integer range
  x, y = n, (n + 1)//2
  while y < x:
    x, y = y, (y + n//y)//2
  return x

def isprime(n): # simple iterative version, faster than non-dynamic sieve for low N, no memory cost, you decide
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

def transpose(matrix):
  return list(map(list, zip(*matrix)))

def lmap(f, *args): # because wrapping in list() all the time is awkward, saves abusing the slow `*a,=map(*args)`!
  return list(map(f, *args))

def tmap(f, *args): # for the versions of python with faster tuple lookups (PEP 590 vectorcalls effect this how?)
  return tuple(map(f, *args))

def join(iterable, sep = " "): # because sep.join(it) doesn't convert str(it) for it in iterable
  return sep.join(map(str, iterable))

def avg(iterable, start=0): # because x/0 = 0 in euclidean
  return sum(iterable, start) / len(iterable) if len(iterable) else 0

def minmax(*iterable): # get the minimum and maximum quickly
  return min(iterable),max(iterable) # min(*iterable) is only faster for len(iterable) == 2

def shuffled(iterable): # aka, "shuffle but not in place", like reversed and sorted, shame they ignore prior
  return sample(iterable, len(iterable))

def lenfilter(iterable, pred=bool): # counts how many are true for a given predicate
  return sum(1 for it in iterable if pred(it))

def first(iterable, default=None): # first item
  return next(iter(iterable), default) 

def dotprod(A, B):
  return sum(a*b for a,b in zip(A,B))

def sample_set(s: set, k: int): # 3.9 was doing so many things right, this is just annoying
  return sample(sorted(s), k) # remember to "sequencify" your sets now bc "reproducibility"

def human_time(t: float, seconds = True): # because nobody seems to get it right
  return f"{int(t//60)}m {human_time((int(t)%60)+(t-int(t)), True)}" if t > 60 else f"{t:.3f}s" if t > 0.1 and seconds else f"{t*1000:.3f}ms" if t > 0.0001 else f"{t*1000000:.3f}us"

if sys.version_info.major >= 3 and sys.version_info.minor >= 10:
  def popcount(x: int):
    return x.bit_count()
else:
  def popcount(x: int):
    return bin(x).count("1")

def bits(x): # because bin() has no option to drop the 0b or sign (shame no uint type)
  return bin(x)[2 if int(x) >= 0 else 3 :]

def find(v, iterable: list, start = 0, stop = -1, missing = -1): # find v in interable without exceptions
  try:
    return iterable.index(v, start, stop)
  except: # because if doesn't have .index then we couldn't find iterable
    return missing
