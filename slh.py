# Santa's Little Helpers
from dataclasses import dataclass
from functools import partial
from itertools import chain
from random import sample
from math import log2
try:
  from math import prod # 3.9
except:
  from functools import reduce
  from operator import mul
  def prod(iterable, start=1): return reduce(mul, iterable, initial=start) # not 3.9 (ie, pypy)

flatten = chain.from_iterable

class dot(dict): # as in a "dot dict", a dict you can access by a "." # pretty inefficient bc. (dict), but elegant
  __getattr__, __setattr__ = dict.__getitem__, dict.__setitem__

@dataclass
class data: # I recommend this pattern in general for POD, slightly more memory with __slots__ but guarantees perf
  __slots__ = ["x", "y", "z", "w"]
  x: float
  y: float
  z: float
  w: float
  def slots(self): # A useful inclusion
    return {slot:self.__getattribute__(slot) for slot in self.__slots__}

def transpose(matrix):
  return list(map(list, zip(*matrix)))

def lmap(f, *args): # because wrapping in list() all the time is awkward, remember to abuse `*a,=map(*args)`!
  return list(map(f, *args))

def tmap(f, *args):
  return tuple(map(f, *args))

def join(iterable, sep = " "): # because sep.join(it) doesn't convert it to an iterable of str
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

def popcount(x: int): # 3.10 adds int.bit_count() so yay
  return bin(x).count("1")

def bits(x): # because bin() has no option to drop the 0b or sign (shame no uint type)
  return bin(x)[2 if int(x) >= 0 else 3 :]

def find(v, iterable: list, start: int, stop: int, missing = -1): # find v in interable without exceptions
  try:
    return iterable.index(v, start, stop)
  except: # because if doesn't have .index then we couldn't find iterable
    return missing
