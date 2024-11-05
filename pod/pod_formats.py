"""
POD Formats.

Copyright 2022 Alex Blandin
"""

from __future__ import annotations

import dataclasses
import sys
from collections import namedtuple
from dataclasses import dataclass
from operator import attrgetter
from typing import NamedTuple

from attrs import define, frozen

if sys.version_info >= (3, 9):
  from functools import cache
else:
  from functools import lru_cache as cache  # pyright: ignore[reportUnreachable]

maybe_slots = {"slots": True} if sys.version_info >= (3, 10) else {}


#################
## POD Formats ##
#################


class Regular:
  """regular class."""

  def __init__(self, sender, receiver, date, amount) -> None:  # noqa: ANN001, D107 # pyright: ignore[reportUnknownParameterType,reportMissingParameterType,reportMissingSuperCall]
    self.sender = sender
    self.receiver = receiver
    self.date = date
    self.amount = amount


class Slots:
  """regular class with slots."""

  __slots__ = ["sender", "amount", "receiver", "date"]

  def __init__(self, sender, receiver, date, amount) -> None:  # noqa: ANN001, D107 # pyright: ignore[reportMissingSuperCall,reportUnknownParameterType,reportMissingParameterType]
    self.sender = sender
    self.receiver = receiver
    self.date = date
    self.amount = amount


NTuple = namedtuple("NTuple", ["sender", "receiver", "date", "amount"])  # noqa: PYI024 # pyright: ignore[reportUntypedNamedTuple,reportAny]
"""named tuple"""

ProcTypedNTuple = NamedTuple("_TypedNTuple", sender=str, receiver=str, date=str, amount=float)  # noqa: UP014
"""typed named tuple, using the procedural kwargs approach"""


class TypedNTuple(NamedTuple):
  """typed named tuple."""

  sender: str
  receiver: str
  date: str
  amount: float


@dataclass
class DataClass:  # RECOMMENDED WHEN YOU CAN'T USE attrs
  """dataclass."""

  sender: str
  receiver: str
  date: str
  amount: float


@define
class AttrClass:  # RECOMMENDED
  """attrs class."""

  sender: str
  receiver: str
  date: str
  amount: float


# if statically type checked, you can use @attrs.define(unsafe_hashes=True) instead of @attrs.frozen
# as in https://threeofwands.com/attra-iv-zero-overhead-frozen-attrs-classes/


@frozen
class FrozenAttrClass:
  """frozen attrs class."""

  sender: str
  receiver: str
  date: str
  amount: float


@define(slots=False)
class AttrUnslot:  # RECOMMENDED
  """attrs class."""

  sender: str
  receiver: str
  date: str
  amount: float


@frozen(slots=False)
class FrozenAttrUnslot:
  """frozen attrs class."""

  sender: str
  receiver: str
  date: str
  amount: float


@dataclass
class DataSlots:
  """dataclass with slots, uses manual entry."""

  __slots__ = ["sender", "amount", "receiver", "date"]
  sender: str
  receiver: str
  date: str
  amount: float


@dataclass(**maybe_slots)
class DataSlotsAuto:
  """dataclass with slots, requires python 3.10+."""

  sender: str
  receiver: str
  date: str
  amount: float


@dataclass(frozen=True)
class FrozenData:
  """frozen dataclass."""

  sender: str
  receiver: str
  date: str
  amount: float


@dataclass(frozen=True)
class FrozenDataSlots:
  """frozen dataclass with slots, uses manual entry."""

  __slots__ = ["sender", "amount", "receiver", "date"]
  sender: str
  receiver: str
  date: str
  amount: float


@dataclass(**maybe_slots, frozen=True)
class FrozenDataSlotsAuto:
  """frozen dataclass with slots, requires python 3.10+."""

  sender: str
  receiver: str
  date: str
  amount: float


@cache
def cls_to_tuple(cls):  # noqa: ANN001, ANN201 # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
  """This converts a class to a NamedTuple; cached because this is expensive!"""
  return NamedTuple(cls.__name__, **cls.__annotations__)  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]


def cls_to_tuple_uncached(cls):  # noqa: ANN001, ANN201 # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
  """This converts a class to a NamedTuple."""
  return NamedTuple(cls.__name__, **cls.__annotations__)  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]


@dataclass(**maybe_slots)
class Struct:
  """a struct-like Plain Old Data base class, this is consistently much faster but breaks when subclassed, use StructSubclassable if you need that."""  # noqa: E501

  sender: str
  receiver: str
  date: str
  amount: float

  def __iter__(self):  # noqa: ANN204 # pyright: ignore[reportUnknownParameterType]
    """Iterating over the values, rather than the __slots__."""
    yield from map(self.__getattribute__, self.__slots__)  # pyright: ignore[reportUnknownArgumentType,reportAttributeAccessIssue,reportUnknownMemberType]

  def __len__(self) -> int:
    """How many slots there are, useful for slices, iteration, and reversing."""
    return len(self.__slots__)  # pyright: ignore[reportUnknownArgumentType,reportAttributeAccessIssue,reportUnknownMemberType]

  def __getitem__(self, n: int | slice):  # noqa: ANN204
    """Generic __slots__[n] -> val, because subscripting (and slicing) is handy at times."""
    if isinstance(n, int):
      return self.__getattribute__(self.__slots__[n])  # pyright: ignore[reportUnknownArgumentType,reportAny,reportAttributeAccessIssue,reportUnknownMemberType]
    else:  # noqa: RET505
      return list(map(self.__getattribute__, self.__slots__[n]))  # pyright: ignore[reportUnknownArgumentType,reportAttributeAccessIssue,reportUnknownMemberType]

  def _astuple(self):  # noqa: ANN202
    """Generic __slots__ -> tuple; super fast, low quality of life."""
    return tuple(map(self.__getattribute__, self.__slots__))  # pyright: ignore[reportUnknownArgumentType,reportAttributeAccessIssue,reportUnknownMemberType]

  def aslist(self):  # noqa: ANN201
    """Generic __slots__ -> list; super fast, low quality of life, a shallow copy."""
    return list(map(self.__getattribute__, self.__slots__))  # pyright: ignore[reportUnknownArgumentType,reportAttributeAccessIssue,reportUnknownMemberType]

  def asdict(self):  # noqa: ANN201 # pyright: ignore[reportUnknownParameterType]
    """Generic __slots__ -> dict; helpful for introspection, limited uses outside debugging."""
    return {slot: self.__getattribute__(slot) for slot in self.__slots__}  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType,reportAttributeAccessIssue,reportUnknownMemberType]

  def astuple(self):  # noqa: ANN201
    """Generic __slots__ -> NamedTuple; a named shallow copy."""
    return cls_to_tuple(type(self))._make(map(self.__getattribute__, self.__slots__))  # pyright: ignore[reportUnknownArgumentType,reportAttributeAccessIssue,reportUnknownMemberType]


@dataclass(**maybe_slots)
class StructSubclassable:
  """a struct-like Plain Old Data base class, we recommend this approach, this has consistently "good" performance and can still be subclassed."""  # noqa: E501

  sender: str
  receiver: str
  date: str
  amount: float

  def __iter__(self):  # noqa: ANN204 # pyright: ignore[reportUnknownParameterType]
    """Iterating over the values, rather than the __slots__."""
    yield from map(self.__getattribute__, self.fields())

  def __len__(self) -> int:
    """How many slots there are, useful for slices, iteration, and reversing."""
    return len(self.fields())

  def __getitem__(self, n: int | slice):  # noqa: ANN204
    """Generic __slots__[n] -> val, because subscripting (and slicing) is handy at times."""
    if isinstance(n, int):
      return self.__getattribute__(self.fields()[n])  # pyright: ignore[reportAny]
    else:  # noqa: RET505
      return list(map(self.__getattribute__, self.fields()[n]))

  def _astuple(self):  # noqa: ANN202
    """Generic __slots__ -> tuple; super fast, low quality of life, a shallow copy."""
    return tuple(map(self.__getattribute__, self.fields()))

  def aslist(self):  # noqa: ANN201
    """Generic __slots__ -> list; super fast, low quality of life, a shallow copy."""
    return list(map(self.__getattribute__, self.fields()))

  def asdict(self):  # noqa: ANN201
    """Generic __slots__ -> dict; helpful for introspection, limited uses outside debugging, a shallow copy."""
    return {slot: self.__getattribute__(slot) for slot in self.fields()}  # pyright: ignore[reportAny]

  def astuple(self):  # noqa: ANN201
    """Generic __slots__ -> NamedTuple; nicer but just slightly slower than asdict."""
    return cls_to_tuple(type(self))._make(map(self.__getattribute__, self.fields()))

  def fields(self):  # noqa: ANN201
    """__slots__ equivalent using the proper fields approach."""
    return list(map(attrgetter("name"), dataclasses.fields(self)))
