"""
short.hand!

Here we collect useful functions and classes, which are often "fast enough".
Star-importing this includes a few handy stdlib imports, great for the REPL.
The addition of Type Parameter Syntax means this only support >= 3.12.

Copyright 2022 Alex Blandin
"""

# Imports used here
import dataclasses
import platform
from collections import ChainMap, defaultdict
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import chain, count
from operator import attrgetter, indexOf, itemgetter
from pathlib import Path
from random import randrange, sample
from time import time

####################
# Import Shorthand #
####################
"""for when `from shorthand import *` is used"""

# ruff: noqa: E402 F401
import itertools
import math
import operator
import os
import re
import sys
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from functools import cache, partial, reduce
from hashlib import file_digest
from itertools import pairwise
from math import prod, sqrt
from typing import Any, Literal, NamedTuple, Self, SupportsIndex

RE_HTTP = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", flags=re.I | re.M | re.U)  # credit @stephenhay

CPYTHON = sys.implementation.name == "cpython"
PYPY = sys.implementation.name == "pypy"

#################
# POD Shorthand #
#################


class Dot(dict):
  """a "dot dict", a dict you can access by a `.` - inefficient vs dataclass, but convenient."""

  __getattr__, __setattr__ = dict.__getitem__, dict.__setitem__  # type: ignore[reportAssignmentType]


@cache
def cls_to_tuple(cls: type) -> type[NamedTuple]:
  """This converts a class to a NamedTuple; cached because this is expensive!"""
  return NamedTuple(cls.__name__, **cls.__annotations__)


# with these POD patterns, the provided methods for iteration and conversion to dicts/tuples are there to aid performance
# as dataclasses.astuple and dataclasses.asdict are an order of magnitude slower even with such a simple class, its deepcopy semantics versus our shallow copies
# this demonstrates a way to easily get great performance while reclaiming quality of life, see the performance testing at the end for an example of usage
# also, as noted since 3.11, we should not be using .__slots__ due to base class issues, but for performance, we do anyway (use dataclasses.fields, see method)


@dataclass
class Struct:
  """a struct-like Plain Old Data base class, this is consistently much faster but breaks when subclassed, use SubStruct if you need that."""

  __slots__ = ()

  def __iter__(self: Self) -> Generator[Any, Any, None]:
    """Iterating over the values, rather than the __slots__."""
    yield from map(self.__getattribute__, self.__slots__)

  def __len__(self: Self) -> int:
    """How many slots there are, useful for slices, iteration, and reversing."""
    return len(self.__slots__)

  def __getitem__(self: Self, name: int | slice) -> Any | list[Any]:
    """Generic __slots__[n] -> val, because subscripting (and slicing) is handy at times."""
    match name:
      case int(__i):
        return self.__getattribute__(self.__slots__[__i])
      case slice() as __s:
        return list(map(self.__getattribute__, self.__slots__[__s]))
      case _:
        raise IndexTypeError

  def __setitem__(self: Self, name: int | slice, value: Any | Iterable[Any]) -> None:
    """Generic __slots__[n] = val, because subscripting (and slicing) is handy at times."""
    match name, value:
      case int(__name), __value:
        self.__setattr__(self.__slots__[__name], __value)
      case slice() as __name, Iterable() as __value:
        list(map(self.__setattr__, self.__slots__[__name], __value))
      case slice(), _:
        raise SliceAssignmentTypeError
      case _:
        raise IndexTypeError

  def _astuple(self: Self) -> tuple[Any, ...]:
    """Generic __slots__ -> tuple; super fast, low quality of life."""
    return tuple(map(self.__getattribute__, self.__slots__))

  def aslist(self: Self) -> list[Any]:
    """Generic __slots__ -> list; super fast, low quality of life, a shallow copy."""
    return list(map(self.__getattribute__, self.__slots__))

  def asdict(self: Self) -> dict[Any, Any]:
    """Generic __slots__ -> dict; helpful for introspection, limited uses outside debugging."""
    return {slot: self.__getattribute__(slot) for slot in self.__slots__}

  def astuple(self: Self) -> NamedTuple:
    """Generic __slots__ -> NamedTuple; a named shallow copy."""
    return cls_to_tuple(type(self))._make(map(self.__getattribute__, self.__slots__))


@dataclass
class SubStruct:
  """a struct-like Plain Old Data base class, we recommend this approach, this has consistently "good" performance and can also be subclassed."""

  def __iter__(self: Self) -> Generator[Any, Any, None]:
    """Iterating over the values, rather than the __slots__."""
    yield from map(self.__getattribute__, self.fields)

  def __len__(self: Self) -> int:
    """How many slots there are, useful for slices, iteration, and reversing."""
    return len(self.fields)

  def __getitem__(self: Self, name: int | slice) -> Any | list[Any]:
    """Generic __slots__[n] -> val, because subscripting (and slicing) is handy at times."""
    match name:
      case int(__i):
        return self.__getattribute__(self.fields[__i])
      case slice() as __s:
        return list(map(self.__getattribute__, self.fields[__s]))
      case _:
        raise IndexTypeError

  def __setitem__(self: Self, name: int | slice, value: Any | Iterable[Any]) -> None:
    """Generic __slots__[n] = val, because subscripting (and slicing) is handy at times."""
    match name, value:
      case int(__name), __value:
        self.__setattr__(self.fields[__name], __value)
      case slice() as __name, Iterable() as __value:
        list(map(self.__setattr__, self.fields[__name], __value))
      case slice(), _:
        raise SliceAssignmentTypeError
      case _:
        raise IndexTypeError

  def _astuple(self: Self) -> tuple[Any, ...]:
    """Generic __slots__ -> tuple; super fast, low quality of life, a shallow copy."""
    return tuple(map(self.__getattribute__, self.fields))

  def aslist(self: Self) -> list[Any]:
    """Generic __slots__ -> list; super fast, low quality of life, a shallow copy."""
    return list(map(self.__getattribute__, self.fields))

  def asdict(self: Self) -> dict[Any, Any]:
    """Generic __slots__ -> dict; helpful for introspection, limited uses outside debugging, a shallow copy."""
    return {slot: self.__getattribute__(slot) for slot in self.fields}

  def astuple(self: Self) -> NamedTuple:
    """Generic __slots__ -> NamedTuple; nicer but just slightly slower than asdict."""
    return cls_to_tuple(type(self))._make(map(self.__getattribute__, self.fields))

  @property
  def fields(self: Self) -> tuple[Any, ...]:
    """__slots__ equivalent using the proper fields approach."""
    return tuple(map(attrgetter("name"), dataclasses.fields(self)))


#######################
# Iterables Shorthand #
#######################

flatten = chain.from_iterable


def head[T](xs: Iterable[T]) -> T:
  """The first item."""
  return next(iter(xs))


def tail[T](xs: Iterable[T]) -> Iterator[T]:
  """Everything but the first item (as an iterable)."""
  ixs = iter(xs)
  _ = next(ixs)
  return ixs


def headtail[T](xs: Iterable[T]) -> tuple[T, Iterator[T]]:
  """The (head, everything else), with everything but the first item as an iterable."""
  ixs = iter(xs)
  return next(ixs), ixs


def groupdict[T, S](xs: Iterable[T], key: Callable[[T], S] | None = None) -> dict[T | S, list[T]]:
  """Make a dict that maps keys and consecutive groups from the iterable.

  Parameters:
  - `xs: Iterable`; Elements to divide into groups according to the key function
  - `key: ((a) -> a) | None = None`; A function for computing the group category for each element. If `None`, the element itself is used for grouping.

  Returns:
  - `dict[a, list[a]]`; Keys mapped to their groups
  """
  d: defaultdict[T | S, list[T]] = defaultdict(list)
  for k, v in itertools.groupby(xs, key=key):
    d[k].extend(list(v))
  return dict(d.items())


class Circular[T](list[T]):
  """Circularly addressable buffer backed by a list, where Circular([0, 1, 2, 3, 4])[-5:10] is [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]."""

  def __getitem__(self: Self, x: int | slice) -> list[Any] | T:
    """...[n] -> val, because subscripting (and slicing) is handy at times."""
    if isinstance(x, slice):
      return [self[i] for i in range(0 if x.start is None else x.start, len(self) if x.stop is None else x.stop, 1 if x.step is None else x.step)]
    return super().__getitem__(x % max(1, len(self)))

  def __setitem__(self: Self, name: int | slice, value: Iterable[T] | Sequence[T] | T) -> None:
    """...[n] = val, because subscripting (and slicing) is handy at times."""
    match name, value:
      case slice() as __s, __value:
        m = max(1, len(self))
        start = name.start or 0
        stop = name.stop or len(self)
        step = name.step or 1
        match start, stop, step:
          case int(), int(), int():
            pass
          case _:
            raise NonIntegerSliceBoundsTypeError
        indices = count(start, step) if name.stop is None else range(start, stop, step)
        try:
          __iter = iter(__value)  # type: ignore[reportArgumentType]
        except TypeError as err:
          raise SliceAssignmentTypeError from err
        for i, v in zip(indices, __iter, strict=False):
          super()[i % m] = v
      case int(__n), __value:
        super()[__n % max(1, len(self))] = __value  # type: ignore[reportCallIssue,reportArgumentType]
      case _:
        raise IndexTypeError

  def repeat(self: Self, times: int | None = None) -> Generator[T, Any, None]:
    """Continue to loop over this circular buffer until you are done."""
    if times is None:
      while True:
        yield from iter(self)
    else:
      for _ in range(times):
        yield from iter(self)


def unique_list(xs: Sequence) -> list:
  """Reduce a list to only its unique elements `[1,1,2,7,2,4] -> [1,2,7,4]`."""
  return list(dict(zip(xs if len(xs) != 1 else xs[0], itertools.repeat(0))))


def unwrap[T, S](f: Callable[[T, T], S], *args: T, **kwargs: T) -> S | None:
  """Because exceptions are bad."""
  with suppress(Exception):
    return f(*args, **kwargs)


def compose[T](*fs: Callable[..., T]) -> Callable[..., T]:
  """Combine each function in fs; evaluates fs[0] first, and fs[-1] last, like fs[-1](fs[-2](...fs[0](*args, **kwargs)...))."""

  def _comp(x: Any) -> T:
    # for the sake of simplicity, it assumes an arity of 1 for every function, because it might want a tuple in, or vargs, who knows
    for f in fs:
      x = f(x)
    return x

  return _comp


def mapcomp[T, S](xs: Iterable[T], *fs: Callable[..., S]) -> Iterator[S] | Iterator[T]:
  """map(compose(*fs), iterable); evaluates fs[0] first, fs[-1] last, so acts like map(fs[-1], map(fs[-2], ... map(fs[0], iterable)...))."""
  it = iter(xs)
  for f in fs:
    it = map(f, it)

  return it


def lmap[T](f: Callable[..., T], *args: Any) -> list[T]:
  """Because wrapping in list() all the time is awkward, saves abusing the slow `*a,=map(*args)`!"""
  return list(map(f, *args))


def transpose[T](matrix: list[list[T]]) -> list[list[T]]:
  """Inefficient but elegant, so if it's a big matrix please don't use."""
  return lmap(list, zip(*matrix, strict=True))


def tmap[T](f: Callable[..., T], *args: Any) -> tuple[T, ...]:
  """For the versions of python with faster tuple lookups."""
  return tuple(map(f, *args))


def join(xs: Iterable, sep: str = " ") -> str:
  """Because sep.join(iterable) doesn't convert to str(i) for i in iterable."""  # noqa: D402
  return sep.join(map(str, xs))


def minmax(xs: Iterable) -> tuple[Any, Any]:
  """Get the minimum and maximum quickly."""
  return min(xs), max(xs)


def minmax_ind(xs: Iterable) -> tuple[tuple[int, Any], tuple[int, Any]]:
  """Minmax but with indices, so ((i_a, min), (i_b, max))."""
  return min(enumerate(xs), key=itemgetter(1)), max(enumerate(xs), key=itemgetter(1))


def shuffled[T](xs: Iterable[T]) -> list[T]:
  """aka, "shuffle but not in place", like reversed() and sorted()."""
  xs = list(xs)  # this way we support sets, without a sort, as sample doesn't anymore
  return sample(xs, len(xs))


def lenfilter[T](xs: Iterable[T], pred: Callable[[T], bool] = bool) -> int:
  """Counts how many are true for a given predicate."""
  return sum(1 for i in xs if pred(i))  # better (esp in pypy) than len(filter()) since not constructing a list


def first[T](xs: Iterable[T], default: T | None = None) -> T | None:
  """The first item in iterable."""
  return next(iter(xs), default)


def sample_set[T](s: set[T], k: int) -> list[T]:
  """Sample a set because you just want some random elements and don't care (about reproducibility)."""
  return sample(list(s), k)  # if you care about reproducibility (with known seeds), sort prior


def sorted_dict_by_key(d: dict, *, reverse: bool = False) -> dict:
  """Sort a dict by key."""
  return dict(sorted(d.items(), key=itemgetter(0), reverse=reverse))


def sorted_dict_by_val(d: dict, *, reverse: bool = False) -> dict:
  """Sort a dict by value."""
  return dict(sorted(d.items(), key=itemgetter(1), reverse=reverse))


def sorted_dict(d: dict, key: Callable = itemgetter(1), *, reverse: bool = False) -> dict:
  """Generic sorting, because it's something people kinda want."""
  return dict(sorted(d.items(), key=key, reverse=reverse))


def sortas[T](first: Iterable[T], second: Iterable) -> list[T]:
  """Sorts the first as if it was the second."""
  return list(map(itemgetter(0), sorted(zip(first, second, strict=True))))


def dedupe[T](it: Iterable[T]) -> Generator[T, Any, None]:
  """Deduplicates an iterator, consumes memory to do so, non-blocking."""
  s = set()
  for i in it:
    if i not in s:
      s.add(i)
      yield i


def find[T](v: T, xs: Iterable[T], start: int | None = None, stop: int | None = None, missing: int | None = -1) -> int | None:
  """Find the first index of v without raising exceptions. WARNING: consumes iterable."""
  with suppress(Exception):
    match xs, start, stop:
      case list(xs), start, stop:
        return xs.index(v, start if start is not None else 0, stop if stop is not None else sys.maxsize)
      case Iterable() as xs, None | 0, None | -1:
        return indexOf(xs, v)
      case Iterable() as xs, start, stop:
        return list(xs).index(v, start or 0, stop or sys.maxsize)
  return missing


class DeepChainMap[K, V](ChainMap[K, V]):
  """Variant of ChainMap that allows direct updates to inner scopes."""

  def __setitem__(self: Self, key: K, value: V) -> None:
    """Deep ...[key] = value by updating the first match, falling back to updating the first map."""
    for mapping in self.maps:
      if key in mapping:
        mapping[key] = value
        return  # we only modify the first match
    self.maps[0][key] = value

  def __delitem__(self: Self, key: K) -> None:
    """Deep del ...[key] by deleting the first match, raising a KeyError if it's not found."""
    for mapping in self.maps:
      if key in mapping:
        del mapping[key]
        return  # we only modify the first match
    raise KeyError(key)


###################
# Maths Shorthand #
###################


def avg(xs: Sequence[float], start: float = 0.0) -> float:
  """No exceptions, because x/0 = 0 in euclidean."""
  return sum(xs, start) / len(xs) if len(xs) else 0


def dotprod(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
  """Quick and dirty dot product if you're not importing numpy."""
  return sum(a * b for a, b in zip(vec_a, vec_b, strict=True))


def bits(x: int) -> str:
  """Because bin() has the annoying 0b, slower but cleaner."""
  return f"{x:b}"


def ilog2(x: int) -> int:
  """Integer log2, aka the position of the first bit."""
  return x.bit_length() - 1


# if you must go faster, use: from gmpy2 import bit_scan1 as ctz
def ctz(v: int) -> int:
  """Count trailing zeroes."""
  return (v & -v).bit_length() - 1


def popcount(x: int) -> int:
  """Number of ones in the binary representation of the absolute value of self."""
  return x.bit_count()


def isqrt(n: int) -> int:
  """Works for all ints, fast for numbers < 2**52 (aka, abusing double precision sqrt)."""
  if n < 2**52:
    return int(sqrt(n))
  n = int(n)
  x, y = n, (n + 1) // 2
  while y < x:
    x, y = y, (y + n // y) // 2
  return x


def isprime(n: int) -> bool:
  """Simple iterative one."""
  if n in {2, 3, 5, 7}:
    return True
  if not (n & 1) or not (n % 3) or not (n % 5) or not (n % 7):
    return False
  if n < 121:
    return n > 1
  sqrt = isqrt(n)
  return all(not (not n % i or not n % (i + 2)) for i in range(11, sqrt, 6))


def fastprime(n: int, trials: int = 8) -> bool:
  """Miller-Rabin primality test.

  - Returns False when n is not prime.
  - Returns True when n is a prime under 3317044064679887385961981, else when n is very likely a prime.

  Increase the number of trials to raise confidence with n >= 3317044064679887385961981 at cost in performance
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

  def witness(a: int) -> bool:
    if pow(a, d, n) == 1:
      return False
    return all(pow(a, 2**i * d, n) != n - 1 for i in range(s))

  if n < 318665857834031151167461:
    b = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]  # covers 64bit
  elif n < 3317044064679887385961981:
    b = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
  else:
    b = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, *[randrange(41, n, 2) for _ in range(trials)]]

  return all(not witness(a) for a in b)


####################
# Timing Shorthand #
####################


def now() -> str:
  """Because sometimes I want the time now."""
  return f"{datetime.now(UTC):%Y-%m-%d-%H-%M-%S}"


def tf[T](func: Callable[..., T], *args: Any, __pretty_tf: bool = True, **kwargs: Any) -> T:
  """Time func func, as in, func to time the function func."""
  start = time()
  r = func(*args, **kwargs)
  end = time()
  if __pretty_tf:
    fargs = list(map(str, (a.__name__ if hasattr(a, "__name__") else a for a in args))) + [f"{k}={v}" for k, v in kwargs.items()]
    print(f"{func.__qualname__}({', '.join(fargs)}) = {r} ({human_time(end - start)})")
  else:
    print(human_time(end - start))
  return r


def human_time(t: float, *, seconds: bool = True) -> str:
  """Because nobody makes it humanly readable."""
  return f"{t // 60:.0f}m {t % 60:.3f}s" if t > 60 else f"{t:.3f}s" if t > 0.1 and seconds else f"{t * 1000:.3f}ms" if t > 0.0001 else f"{t * 10**6:.3f}us"


def hours_minutes_seconds(t: float) -> None:
  """From some number of seconds t, how many (years) (weeks) (days) (hours) minutes and seconds are there (filled in as needed)."""
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


def yesno(
  prompt: str = "", *, accept_return_as: bool | None = None, replace_lists: bool = False, yes_group: set[str] | None = None, no_group: set[str] | None = None
) -> bool:
  """Keep asking until they say yes or no (if accept_return_as is True then ENTER/RETURN is a yes, if accept_return_as is False than it's a no)."""
  if no_group is None:
    no_group = set()
  if yes_group is None:
    yes_group = set()
  msg = f"{prompt} [Y/n]: " if accept_return_as else f"{prompt} [y/n]: " if accept_return_as is None else f"{prompt} [y/N]: "
  while True:
    reply = input(msg).strip().lower()
    if reply in (yes_group if replace_lists else {"y", "ye", "yes"} | yes_group) or (accept_return_as and not reply):
      return True
    if reply in (no_group if replace_lists else {"n", "no"} | no_group) or accept_return_as is False:
      return False


# these to/from bytes wrappers are just for dunder "ephemeral" bytes, use normal int.to/from when byteorder matters
def to_bytes(x: int, nbytes: int | None = None, *, signed: bool | None = None, byteorder: Literal["little", "big"] = sys.byteorder) -> bytes:
  """int.to_bytes but with (sensible) default values, by default assumes unsigned if >=0, signed if <0."""
  return x.to_bytes((nbytes or (x.bit_length() + 7) // 8), byteorder, signed=(x >= 0) if signed is None else signed)


def from_bytes(b: bytes, *, signed: bool = False, byteorder: Literal["little", "big"] = sys.byteorder) -> int:
  """int.from_bytes but sensible byteorder, you must say if it's signed."""
  return int.from_bytes(b, byteorder, signed=signed)


##################
# Path Shorthand #
##################

# convenience functions to not write as much


def resolve(path: str | Path) -> Path:
  """Resolve a Path including "~" (bc Path(path) doesn't...)."""
  return Path(path).expanduser()


@cache
def filedigest(path: str | Path, h: str = "sha1") -> str:
  """Fingerprint a file, caches so modified files bypass with filedigest.__wrapped__ or filedigest.cache_clear()."""
  with Path(path).open("rb") as f:
    return file_digest(f, h).hexdigest()


def readlines(fp: str | Path, encoding: str = "utf8") -> list[str]:
  """Just reads lines as you normally would want to."""
  return resolve(fp).read_text(encoding).splitlines()


def readlinesmap[T](fp: str | Path, *fs: Callable[..., T], encoding: str = "utf8") -> list[T]:
  """Readlines but map each function in fs to fp's lines in order (fs[0]: first, ..., fs[-1]: last)."""
  return list(mapcomp(resolve(fp).read_text(encoding).splitlines(), *fs))  # type: ignore[reportReturnType]


def writelines(fp: str | Path, lines: str | list[str], encoding: str = "utf8", newline: str = "\n") -> int:
  """Just writes lines as you normally would want to."""
  return resolve(fp).write_text(lines if isinstance(lines, str) else newline.join(lines), encoding=encoding, newline=newline)


def writelinesmap(fp: str | Path, lines: str | list[str], *fs: Callable, encoding: str = "utf8", newline: str = "\n") -> int:
  """Writelines but map each function in fs to fp's lines in order (fs[0] first, fs[-1] last)."""
  return resolve(fp).write_text(newline.join(mapcomp(lines if isinstance(lines, list) else lines.splitlines()), *fs), encoding=encoding, newline=newline)


####################
# String Shorthand #
####################


def lev(s1: str, s2: str) -> int:
  """Calculate Levenshtein (edit) distance between strings."""
  if s1 == s2:
    return 0
  l1, l2 = len(s1), len(s2)
  if 0 in (l1, l2):  # noqa: PLR6201 # `tuple[int,int]` is slightly faster than `set[int]` for .__contains__(int)
    return l1 or l2
  if l1 > l2:
    s1, s2, l1, l2 = s2, s1, l2, l1
  d0, d1 = list(range(l2 + 1)), list(range(l2 + 1))
  for i, x in enumerate(s1, 1):
    d1[0] = i
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


########################
# Exceptions Shorthand #
########################


class NonIntegerSliceBoundsTypeError(TypeError):
  """Slice values must be integers."""

  def __init__(self: Self) -> None:  # noqa: D107
    super().__init__(self.__doc__)


class SliceAssignmentTypeError(TypeError):
  """When assigning to a slice, the assigned values must be provided in an interable or sequence."""

  def __init__(self: Self) -> None:  # noqa: D107
    super().__init__(self.__doc__)


class IndexTypeError(TypeError):
  """Inappropriate index type. You can only index using integers or slice objects."""

  def __init__(self: Self) -> None:  # noqa: D107
    super().__init__(self.__doc__)


#########################
# Performance & Testing #
#########################

if __name__ == "__main__":
  from dataclasses import asdict, astuple
  from timeit import repeat

  N_RUNS, N_ITERATIONS = 10, 10**6

  # ruff: noqa: D101,D105
  @dataclass(slots=True)
  class Vec4Struct(Struct):
    x: float
    y: float
    z: float
    w: float

  @dataclass(slots=True)
  class Vec4SubStruct(SubStruct):
    x: float
    y: float
    z: float
    w: float

  @dataclass(slots=True)
  class BestTime:
    cpython: float | None = None
    pypy: float | None = None

    def log_time(self: Self, x: float) -> None:
      """If this was an improvement, log it."""
      if CPYTHON:
        self.cpython = min(self.cpython, x) if self.cpython else x
      elif PYPY:
        self.pypy = min(self.pypy, x) if self.pypy else x
      else:
        raise NotImplementedError

    @classmethod
    def new(cls, x: float) -> Self:  # noqa: ANN102
      """Create a time from x."""
      self = cls()
      if CPYTHON:
        self.cpython = x
      elif PYPY:
        self.pypy = x
      else:
        raise NotImplementedError
      return self

    def __repr__(self: Self) -> str:
      return f"BestTime({self.cpython:.6f}, {self.pypy:.6f})"

    def __str__(self: Self) -> str:
      return human_time(self.cpython or self.pypy or 10.0) if self.cpython or self.pypy else "??????"

  @dataclass(slots=True)
  class TimingRow:
    code: str
    struct: BestTime
    substruct: BestTime
    comment: str

    def __repr__(self: Self) -> str:
      return f'    TimingRow("{self.code}", {self.struct!r}, {self.substruct!r}, "{self.comment}"),'

    def __str__(self: Self) -> str:
      return f"{self.code} | {self.struct} \t | {self.substruct} \t | {self.comment}"

  # python -m pip install -U py-cpuinfo
  # If I really want portability, then hey, there it is
  # But for now, I can just use my own table of my devices

  table = defaultdict(platform.processor)  # otherwise just report that back
  table.update(
    {
      "Intel64 Family 6 Model 158 Stepping 10, GenuineIntel": "Intel 8700k",
      "AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD": "AMD 7980HS",
    },
  )
  CPU = table[platform.processor()]
  device = f"{platform.node()} (ROG Flow X13) w/ {CPU}" if CPU == "AMD 7980HS" else f"{platform.node()} w/ {CPU}"

  print(sys.version)
  print(f"Benchmarked on {device} using {'CPython' if CPYTHON else 'PyPy' if PYPY else 'Python'}")
  print(f"Lowest time over {N_RUNS} runs of {N_ITERATIONS} iterations of each microbench: ")

  tests = [  # best times for N_ITERATIONS = 10**6 is still on Asteria (ROG Flow X13) w/ 7980HS using CPython 3.12.0, PyPy 3.10.13/7.3.13
    TimingRow("tuple(d)      ", BestTime(0.314000, 0.095916), BestTime(1.586000, 0.249744), "a tuple around .__iter__"),
    TimingRow("astuple(d)    ", BestTime(0.978000, 0.330000), BestTime(0.983568, 0.345000), "a dataclasses.astuple"),
    TimingRow("d._astuple()  ", BestTime(0.209000, 0.065952), BestTime(0.836000, 0.227647), "a shallow copy dataclasses.astuple"),
    TimingRow("d.astuple()   ", BestTime(0.398000, 0.154000), BestTime(1.077000, 0.328403), "a namedtuple d._astuple()"),
    TimingRow("asdict(d)     ", BestTime(0.984542, 0.447809), BestTime(0.982506, 0.456000), "a dataclasses.asdict"),
    TimingRow("d.asdict()    ", BestTime(0.289112, 0.105000), BestTime(0.983000, 0.267000), "a shallow copy dataclasses.asdict"),
    TimingRow("d.x           ", BestTime(0.010484, 0.000585), BestTime(0.010416, 0.000585), "a typical operator"),
    TimingRow("d[0]          ", BestTime(0.083811, 0.000594), BestTime(0.708000, 0.155892), "a typical operator"),
    TimingRow("d[-1]         ", BestTime(0.088118, 0.000591), BestTime(0.723000, 0.154042), "a typical operator"),
    TimingRow("d[:1]         ", BestTime(0.194000, 0.025000), BestTime(0.832000, 0.170620), "a typical operator"),
    TimingRow("d[:]          ", BestTime(0.260000, 0.057000), BestTime(0.926000, 0.225085), "a typical operator"),
    TimingRow("list(d)       ", BestTime(0.352000, 0.080535), BestTime(1.617000, 0.386774), "much slower than the [:] operator"),
    TimingRow("d.aslist()    ", BestTime(0.199000, 0.055000), BestTime(0.812000, 0.212589), "comparable to the [:] operator"),
    TimingRow("d[::-1]       ", BestTime(0.263000, 0.061000), BestTime(0.910000, 0.223778), "a typical operator"),
    TimingRow("d.x = 0.1     ", BestTime(0.010197, 0.000591), BestTime(0.010249, 0.000588), "a typical operator"),
    TimingRow("d[0] = 0.1    ", BestTime(0.114828, 0.000599), BestTime(0.839096, 0.154319), "a typical operator"),
    TimingRow("d[0:1] = [0.1]", BestTime(0.454736, 0.069207), BestTime(1.222534, 0.219911), "a typical operator"),
  ]

  base_data = [1.2, 3.4, 5.6, 7.8]
  print("code           | struct \t | substruct \t | comment")
  for test in tests:
    struct_time = min(repeat(test.code, "d = Vec4Struct(*base_data)", number=N_ITERATIONS, repeat=N_RUNS, globals=globals()))
    substruct_time = min(repeat(test.code, "d = Vec4SubStruct(*base_data)", number=N_ITERATIONS, repeat=N_RUNS, globals=globals()))
    print(TimingRow(test.code, BestTime.new(struct_time), BestTime.new(substruct_time), test.comment), flush=True)
    test.struct.log_time(struct_time)
    test.substruct.log_time(substruct_time)
  print()
  print("The best times now stand at:")
  for test in tests:
    print(repr(test))
