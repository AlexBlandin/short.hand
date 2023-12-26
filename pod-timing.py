"""
██████╗ ██╗      █████╗ ██╗███╗   ██╗
██╔══██╗██║     ██╔══██╗██║████╗  ██║
██████╔╝██║     ███████║██║██╔██╗ ██║
██╔═══╝ ██║     ██╔══██║██║██║╚██╗██║
██║     ███████╗██║  ██║██║██║ ╚████║
╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝

 ██████╗ ██╗     ██████╗
██╔═══██╗██║     ██╔══██╗
██║   ██║██║     ██║  ██║
██║   ██║██║     ██║  ██║
╚██████╔╝███████╗██████╔╝
 ╚═════╝ ╚══════╝╚═════╝

██████╗  █████╗ ████████╗ █████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
██║  ██║███████║   ██║   ███████║
██║  ██║██╔══██║   ██║   ██╔══██║
██████╔╝██║  ██║   ██║   ██║  ██║
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝

for when you can't use attrs
"""

import contextlib
import dataclasses
import timeit
from collections import namedtuple
from dataclasses import dataclass
from functools import cache
from operator import attrgetter
from statistics import geometric_mean, harmonic_mean, mean, median, pstdev, pvariance, quantiles, stdev, variance
from typing import NamedTuple

try:
  from sys import getsizeof

  getsizeof("sorry pypy, but cry about it, I want the overhead, not the recursive size")
except Exception:

  def getsizeof(obj: object, default: int = -1) -> int:
    return -1


#########
# Dials #
#########

N_ITERATIONS, N_RUNS = 10**6, 10
PRINTSTATS = False  # only need to print out when there's something new

##############
# Formatting #
##############

print(f"POD test results for {N_ITERATIONS} iterations, best of {N_RUNS} runs:")
sep = f"+-{"":->23}-+-{"":->4}-+-{"":->9}-+-{"":->9}-+"
print(sep)
print(f"| {"name":>23} | size | make (ms) | item (ms) |")
print(sep)


def time(code: str, iterations: int = N_ITERATIONS, runs: int = N_RUNS, setup: str = ""):
  """time how long something takes, result is min recorded time for an entire run in seconds"""
  return min(timeit.repeat(code, setup=setup, number=iterations, repeat=runs, globals=globals()))


def row(name: str, new: str, access: str):
  """another row in the table"""
  # 1000* gives ms
  with contextlib.suppress(Exception):
    print(f"| {name:>23} | {getsizeof(eval(new)):04} | {1000 * time(new): >9.4f} | {1000 * time("var" + access, setup = "var=" + new): >9.4f} |")


#################
# Test Subjects #
#################


class Regular:
  """regular class"""

  def __init__(self, sender, receiver, date, amount):
    self.sender = sender
    self.receiver = receiver
    self.date = date
    self.amount = amount


class Slots:
  """regular class with slots"""

  __slots__ = ["sender", "amount", "receiver", "date"]

  def __init__(self, sender, receiver, date, amount):
    self.sender = sender
    self.receiver = receiver
    self.date = date
    self.amount = amount


NTuple = namedtuple("NTuple", ["sender", "receiver", "date", "amount"])
"""named tuple"""

ProcTypedNTuple = NamedTuple("_TypedNTuple", sender=str, receiver=str, date=str, amount=float)  # noqa: UP014
"""typed named tuple, using the procedural kwargs approach"""


class TypedNTuple(NamedTuple):
  """typed named tuple"""

  sender: str
  receiver: str
  date: str
  amount: float


@dataclass
class DataClass:  # RECOMMENDED WHEN YOU CAN'T USE attrs
  """dataclass"""

  sender: str
  receiver: str
  date: str
  amount: float


@dataclass
class DataSlots:
  """dataclass with slots, uses manual entry"""

  __slots__ = ["sender", "amount", "receiver", "date"]
  sender: str
  receiver: str
  date: str
  amount: float


@dataclass(slots=True)
class DataSlotsAuto:
  """dataclass with slots, requires python 3.10+"""

  sender: str
  receiver: str
  date: str
  amount: float


@dataclass(frozen=True)
class FrozenData:
  """frozen dataclass"""

  sender: str
  receiver: str
  date: str
  amount: float


@dataclass(frozen=True)
class FrozenDataSlots:
  """frozen dataclass with slots, uses manual entry"""

  __slots__ = ["sender", "amount", "receiver", "date"]
  sender: str
  receiver: str
  date: str
  amount: float


@dataclass(slots=True, frozen=True)
class FrozenDataSlotsAuto:
  """frozen dataclass with slots, requires python 3.10+"""

  sender: str
  receiver: str
  date: str
  amount: float


@cache
def cls_to_tuple(cls):
  """this converts a class to a NamedTuple; cached because this is expensive!"""
  return NamedTuple(cls.__name__, **cls.__annotations__)


@dataclass(slots=True)
class Struct:
  """a struct-like Plain Old Data base class, this is consistently much faster but breaks when subclassed, use StructSubclassable if you need that"""

  sender: str
  receiver: str
  date: str
  amount: float

  def __iter__(self):
    """iterating over the values, rather than the __slots__"""
    yield from map(self.__getattribute__, self.__slots__)  # type: ignore

  def __len__(self):
    """how many slots there are, useful for slices, iteration, and reversing"""
    return len(self.__slots__)  # type: ignore

  def __getitem__(self, n: int | slice):
    """generic __slots__[n] -> val, because subscripting (and slicing) is handy at times"""
    if isinstance(n, int):
      return self.__getattribute__(self.__slots__[n])  # type: ignore
    else:
      return list(map(self.__getattribute__, self.__slots__[n]))  # type: ignore

  def _astuple(self):
    """generic __slots__ -> tuple; super fast, low quality of life"""
    return tuple(map(self.__getattribute__, self.__slots__))  # type: ignore

  def aslist(self):
    """generic __slots__ -> list; super fast, low quality of life, a shallow copy"""
    return list(map(self.__getattribute__, self.__slots__))  # type: ignore

  def asdict(self):
    """generic __slots__ -> dict; helpful for introspection, limited uses outside debugging"""
    return {slot: self.__getattribute__(slot) for slot in self.__slots__}  # type: ignore

  def astuple(self):
    """generic __slots__ -> NamedTuple; a named shallow copy"""
    return cls_to_tuple(type(self))._make(map(self.__getattribute__, self.__slots__))  # type: ignore


@dataclass(slots=True)
class StructSubclassable:
  """a struct-like Plain Old Data base class, we recommend this approach, this has consistently "good" performance and can still be subclassed"""

  sender: str
  receiver: str
  date: str
  amount: float

  def __iter__(self):
    """iterating over the values, rather than the __slots__"""
    yield from map(self.__getattribute__, self.fields())

  def __len__(self):
    """how many slots there are, useful for slices, iteration, and reversing"""
    return len(self.fields())

  def __getitem__(self, n: int | slice):
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


PODS = [
  Regular,
  Slots,
  NTuple,
  ProcTypedNTuple,
  TypedNTuple,
  DataClass,
  DataSlots,
  DataSlotsAuto,
  FrozenData,
  FrozenDataSlots,
  FrozenDataSlotsAuto,
  Struct,
  StructSubclassable,
]

#########
# Tests #
#########

row("dict from literal", "{'amount': 1.0, 'receiver': 'me', 'date': '2022-01-01', 'sender': 'you'}", "['receiver']")
row("tuple from literal", "(1.0, 'me', '2022-01-01', 'you')", "[1]")
row("regular class", "Regular(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("class using slots", "Slots(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("collections namedtuple", "NTuple(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("proc. typed NamedTuple", "ProcTypedNTuple(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("typed NamedTuple", "TypedNTuple(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("regular dataclass", "DataClass(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("regular dataclass slots", "DataSlots(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("autogen dataclass slots", "DataSlotsAuto(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("frozen dataclass", "FrozenData(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("frozen dataclass slots", "FrozenDataSlots(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("autogen frozen dc slots", "FrozenDataSlotsAuto(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("handier dataclass slots", "Struct(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("even handier dataclasss", "StructSubclassable(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")

print(sep)
print()

###########
# Results #
###########

# 8700k is an Intel 8700k equipped desktop
# 95 W, 3.7 GHz base, clocked at 4.3 GHz, turbo disabled
# 4x8 GB 3000 MT/s 16-20-20-38 DDR4

# 4700U is a Ryzen 4700U equipped laptop (HP Envy x360 13")
# 15 W base, 20 W turbo, 2.0 GHz base, 3.8-3.9 GHz stable during testing, 4.1 GHz turbo
# 2x8 GB 3200 MT/s 22-22-22-52 DDR4
# running in HP's "recommended" mode (not "performance" or "quiet"), so I've included battery too

# 7980HS is a Ryzen 7980HS equipped laptop (ROG Flow X13)
# 31 W base, 31 W turbo, 4.0 GHz base, 5.2 GHz turbo
# 2x8 GB 6400 MT/s 19-15-17-34 LPDDR5
# running in the Windows 11 "performance" mode for this

# ruff: noqa: N816

win_cpython_3_12_7980HS_plugged = """
POD test results for 1000000 iterations, best of 10 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | 0184 |   55.5360 |   12.7678 |
|      tuple from literal | 0072 |    7.9822 |   11.9768 |
|           regular class | 0048 |  173.2675 |   11.3303 |
|       class using slots | 0064 |  167.3380 |   11.2800 |
|  collections namedtuple | 0072 |  222.3365 |   15.3873 |
|  proc. typed NamedTuple | 0072 |  225.7671 |   15.1912 |
|        typed NamedTuple | 0072 |  219.2748 |   15.0950 |
|       regular dataclass | 0048 |  173.7466 |   11.2866 |
| regular dataclass slots | 0064 |  166.7982 |   11.2796 |
| autogen dataclass slots | 0064 |  168.5132 |   11.4792 |
|        frozen dataclass | 0048 |  411.0357 |   11.4162 |
|  frozen dataclass slots | 0064 |  403.9585 |   11.2933 |
| autogen frozen dc slots | 0064 |  415.2466 |   11.3032 |
| handier dataclass slots | 0064 |  170.6867 |   11.3395 |
| even handier dataclasss | 0064 |  168.9459 |   11.2819 |
+-------------------------+------+-----------+-----------+
"""

win_cpython_3_10_7980HS_plugged = """
POD test results for 1000000 iterations, best of 10 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | 0232 |   53.5189 |   13.1514 |
|      tuple from literal | 0072 |    4.6460 |   13.7727 |
|           regular class | 0048 |  209.3287 |   14.8201 |
|       class using slots | 0064 |  189.9015 |   14.0847 |
|  collections namedtuple | 0072 |  211.6640 |   14.2420 |
|  proc. typed NamedTuple | 0072 |  225.8740 |   15.8886 |
|        typed NamedTuple | 0072 |  212.8599 |   14.0693 |
|       regular dataclass | 0048 |  216.5925 |   14.9098 |
| regular dataclass slots | 0064 |  187.5435 |   14.1036 |
| autogen dataclass slots | 0064 |  185.9464 |   14.1274 |
|        frozen dataclass | 0048 |  368.1642 |   14.8384 |
|  frozen dataclass slots | 0064 |  342.5285 |   14.1245 |
| autogen frozen dc slots | 0064 |  338.4191 |   14.5742 |
| handier dataclass slots | 0064 |  187.2570 |   14.0756 |
| even handier dataclasss | 0064 |  185.0936 |   20.7910 |
+-------------------------+------+-----------+-----------+
"""

win_cpython_3_10_4700U_plugged = """
POD test results for 1000000 iterations, best of 100 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | 0232 |   77.4804 |   24.0198 |
|      tuple from literal | 0072 |    7.7768 |   24.3028 |
|           regular class | 0048 |  419.9600 |   22.1962 |
|       class using slots | 0064 |  370.3981 |   21.2082 |
|  collections namedtuple | 0072 |  404.1549 |   19.9666 |
|  proc. typed NamedTuple | 0072 |  406.0698 |   19.7323 |
|        typed NamedTuple | 0072 |  402.0191 |   20.0104 |
|       regular dataclass | 0048 |  417.8476 |   22.4429 |
| regular dataclass slots | 0064 |  367.6712 |   21.0758 |
| autogen dataclass slots | 0064 |  377.8337 |   21.2881 |
|        frozen dataclass | 0048 |  794.3493 |   22.7376 |
|  frozen dataclass slots | 0064 |  752.0625 |   21.3258 |
| autogen frozen dc slots | 0064 |  760.5823 |   21.7042 |
+-------------------------+------+-----------+-----------+
"""

win_cpython_3_10_4700U_battery = """
POD test results for 1000000 iterations, best of 10 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | 0232 |  129.8398 |   46.1178 |
|      tuple from literal | 0072 |   13.1828 |   51.6418 |
|           regular class | 0048 |  775.1731 |   44.0287 |
|       class using slots | 0064 |  461.8491 |   26.3737 |
|  collections namedtuple | 0072 |  463.4976 |   20.9656 |
|  proc. typed NamedTuple | 0072 |  468.3234 |   19.5120 |
|        typed NamedTuple | 0072 |  449.3548 |   19.7859 |
|       regular dataclass | 0048 |  459.1070 |   22.4903 |
| regular dataclass slots | 0064 |  424.1747 |   21.3831 |
| autogen dataclass slots | 0064 |  452.0948 |   28.5871 |
|        frozen dataclass | 0048 | 1191.4967 |   37.7340 |
|  frozen dataclass slots | 0064 |  901.3104 |   21.8362 |
| autogen frozen dc slots | 0064 |  843.3253 |   23.4631 |
+-------------------------+------+-----------+-----------+
"""

win_cpython_3_10_8700k = """
POD test results for 1000000 iterations, best of 100 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | 0232 |   85.1526 |   22.5792 |
|      tuple from literal | 0072 |    8.2798 |   20.2340 |
|           regular class | 0048 |  393.2533 |   23.0611 |
|       class using slots | 0064 |  341.9360 |   22.6880 |
|  collections namedtuple | 0072 |  358.9367 |   20.7772 |
|  proc. typed NamedTuple | 0072 |  358.3562 |   23.3201 |
|        typed NamedTuple | 0072 |  361.5343 |   21.2919 |
|       regular dataclass | 0048 |  395.0910 |   24.0210 |
| regular dataclass slots | 0064 |  345.4267 |   21.2310 |
| autogen dataclass slots | 0064 |  348.0640 |   22.3947 |
|        frozen dataclass | 0048 |  746.4775 |   23.1717 |
|  frozen dataclass slots | 0064 |  677.4594 |   20.7030 |
| autogen frozen dc slots | 0064 |  679.9086 |   21.2003 |
+-------------------------+------+-----------+-----------+
"""

deb_cpython_3_9_8700k = """
POD test results for 1000000 iterations, best of 100 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | 0232 |   68.9289 |   16.5077 |
|      tuple from literal | 0072 |    5.6050 |   15.3206 |
|           regular class | 0048 |  321.4695 |   17.6356 |
|       class using slots | 0064 |  292.5633 |   20.0772 |
|  collections namedtuple | 0072 |  294.4185 |   17.1632 |
|  proc. typed NamedTuple | 0072 |  296.6719 |   17.6866 |
|        typed NamedTuple | 0072 |  294.3780 |   17.1658 |
|       regular dataclass | 0048 |  321.2171 |   18.1571 |
| regular dataclass slots | 0064 |  291.5041 |   18.7349 |
|        frozen dataclass | 0048 |  634.3641 |   17.6353 |
|  frozen dataclass slots | 0064 |  594.0366 |   20.1110 |
+-------------------------+------+-----------+-----------+
"""

deb_cpython_3_9_7980HS_plugged = """
POD test results for 1000000 iterations, best of 10 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | 0232 |   44.2386 |   15.3079 |
|      tuple from literal | 0072 |    3.6364 |   10.5990 |
|           regular class | 0048 |  181.3483 |   14.5877 |
|       class using slots | 0064 |  163.5852 |   14.4710 |
|  collections namedtuple | 0072 |  162.9899 |   13.0975 |
|  proc. typed NamedTuple | 0072 |  162.5954 |   13.1346 |
|        typed NamedTuple | 0072 |  166.5357 |   13.1450 |
|       regular dataclass | 0048 |  186.0436 |   14.6888 |
| regular dataclass slots | 0064 |  164.2548 |   14.8198 |
| autogen dataclass slots | 0064 |  163.3472 |   14.5332 |
|        frozen dataclass | 0048 |  358.7110 |   14.4998 |
|  frozen dataclass slots | 0064 |  352.9367 |   14.5911 |
| autogen frozen dc slots | 0064 |  349.0478 |   14.6551 |
| handier dataclass slots | 0048 |  180.8845 |   14.6661 |
| even handier dataclasss | 0048 |  184.8375 |   14.5750 |
+-------------------------+------+-----------+-----------+
"""

deb_cpython_3_9_4700U_plugged = """
POD test results for 1000000 iterations, best of 10 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | 0232 |   67.4939 |   16.3071 |
|      tuple from literal | 0072 |    4.8811 |   14.7222 |
|           regular class | 0048 |  343.6451 |   19.4977 |
|       class using slots | 0064 |  314.6168 |   20.3658 |
|  collections namedtuple | 0072 |  324.5867 |   18.5107 |
|  proc. typed NamedTuple | 0072 |  325.6120 |   18.3033 |
|        typed NamedTuple | 0072 |  325.4169 |   18.2874 |
|       regular dataclass | 0048 |  344.2590 |   19.5564 |
| regular dataclass slots | 0064 |  316.9539 |   20.3715 |
|        frozen dataclass | 0048 |  697.5164 |   19.4942 |
|  frozen dataclass slots | 0064 |  645.1800 |   20.6960 |
+-------------------------+------+-----------+-----------+
"""

deb_cpython_3_9_4700U_battery = """
POD test results for 1000000 iterations, best of 10 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | 0232 |  114.3563 |   28.1532 |
|      tuple from literal | 0072 |    8.4461 |   26.9623 |
|           regular class | 0048 |  640.5819 |   31.9918 |
|       class using slots | 0064 |  577.8327 |   32.3710 |
|  collections namedtuple | 0072 |  493.8302 |   33.8196 |
|  proc. typed NamedTuple | 0072 |  348.2305 |   18.1998 |
|        typed NamedTuple | 0072 |  342.9820 |   19.0184 |
|       regular dataclass | 0048 |  393.6214 |   21.3467 |
| regular dataclass slots | 0064 |  370.9233 |   21.2177 |
|        frozen dataclass | 0048 |  861.5859 |   34.6186 |
|  frozen dataclass slots | 0064 | 1050.7468 |   29.0954 |
+-------------------------+------+-----------+-----------+
"""

win_pypy_3_10_13_7980HS_plugged = """
POD test results for 1000000 iterations, best of 10 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | -001 |    0.3953 |    8.9181 |
|      tuple from literal | -001 |    0.3921 |    0.3961 |
|           regular class | -001 |    0.3948 |    0.3963 |
|       class using slots | -001 |    0.3968 |    0.3926 |
|  collections namedtuple | -001 |    0.3914 |    0.4284 |
|  proc. typed NamedTuple | -001 |    0.3913 |    0.3908 |
|        typed NamedTuple | -001 |    0.3953 |    0.3967 |
|       regular dataclass | -001 |    0.3970 |    0.3958 |
| regular dataclass slots | -001 |    0.3958 |    0.3959 |
| autogen dataclass slots | -001 |    0.3955 |    0.3947 |
|        frozen dataclass | -001 |    0.3955 |    0.3940 |
|  frozen dataclass slots | -001 |    0.3961 |    0.3976 |
| autogen frozen dc slots | -001 |    0.3991 |    0.3979 |
| handier dataclass slots | -001 |    0.3981 |    0.3970 |
| even handier dataclasss | -001 |    0.3976 |    0.3977 |
+-------------------------+------+-----------+-----------+
"""

win_pypy_3_9_8700k = """
POD test results for 1000000 iterations, best of 100 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | 0000 |    0.6991 |   13.5548 |
|      tuple from literal | 0000 |    0.5326 |    0.6842 |
|           regular class | 0000 |    0.5349 |    0.6003 |
|       class using slots | 0000 |    0.5353 |    0.5346 |
|  collections namedtuple | 0000 |    0.5343 |    0.6852 |
|  proc. typed NamedTuple | 0000 |    0.6858 |    0.5335 |
|        typed NamedTuple | 0000 |    0.5342 |    0.6834 |
|       regular dataclass | 0000 |    0.5354 |    0.6847 |
| regular dataclass slots | 0000 |    0.5357 |    0.5351 |
|        frozen dataclass | 0000 |    0.5369 |    0.6849 |
|  frozen dataclass slots | 0000 |    0.5372 |    0.5815 |
+-------------------------+------+-----------+-----------+
"""

win_pypy_3_10_7980HS_plugged = """
POD test results for 1000000 iterations, best of 10 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | -001 |    0.3984 |    9.9087 |
|      tuple from literal | -001 |    0.3959 |    0.3963 |
|           regular class | -001 |    0.4044 |    0.3965 |
|       class using slots | -001 |    0.4023 |    0.3993 |
|  collections namedtuple | -001 |    0.4017 |    0.4098 |
|  proc. typed NamedTuple | -001 |    0.3952 |    0.3945 |
|        typed NamedTuple | -001 |    0.3971 |    0.4016 |
|       regular dataclass | -001 |    0.3959 |    0.3978 |
| regular dataclass slots | -001 |    0.3965 |    0.3992 |
| autogen dataclass slots | -001 |    0.4010 |    0.3985 |
|        frozen dataclass | -001 |    0.3993 |    0.3949 |
|  frozen dataclass slots | -001 |    0.4005 |    0.4006 |
| autogen frozen dc slots | -001 |    0.3991 |    0.4011 |
| handier dataclass slots | -001 |    0.3973 |    0.4049 |
| even handier dataclasss | -001 |    0.3962 |    0.3962 |
+-------------------------+------+-----------+-----------+
"""

win_pypy_3_9_4700U_plugged = """
POD test results for 1000000 iterations, best of 10000 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | 0000 |    0.5567 |   13.4704 |
|      tuple from literal | 0000 |    0.5622 |    0.5600 |
|           regular class | 0000 |    0.5567 |    0.5568 |
|       class using slots | 0000 |    0.5614 |    0.7157 |
|  collections namedtuple | 0000 |    0.5600 |    0.5600 |
|  proc. typed NamedTuple | 0000 |    0.5567 |    0.5567 |
|        typed NamedTuple | 0000 |    0.5567 |    0.5570 |
|       regular dataclass | 0000 |    0.5567 |    0.5567 |
| regular dataclass slots | 0000 |    0.5680 |    0.5658 |
|        frozen dataclass | 0000 |    0.5567 |    0.5568 |
|  frozen dataclass slots | 0000 |    0.5567 |    0.5567 |
+-------------------------+------+-----------+-----------+
"""

win_pypy_3_9_4700U_battery = """
POD test results for 1000000 iterations, best of 10 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | 0000 |    0.9427 |   25.8458 |
|      tuple from literal | 0000 |    0.9630 |    0.9480 |
|           regular class | 0000 |    0.9559 |    1.2427 |
|       class using slots | 0000 |    0.9464 |    0.9418 |
|  collections namedtuple | 0000 |    0.9443 |    0.9432 |
|  proc. typed NamedTuple | 0000 |    0.9441 |    0.9437 |
|        typed NamedTuple | 0000 |    0.9447 |    1.1097 |
|       regular dataclass | 0000 |    1.6885 |    1.2091 |
| regular dataclass slots | 0000 |    1.7018 |    0.9422 |
|        frozen dataclass | 0000 |    3.6359 |    2.1674 |
|  frozen dataclass slots | 0000 |    0.9501 |    0.9422 |
+-------------------------+------+-----------+-----------+
"""

deb_pypy_3_9_8700k = """
POD test results for 1000000 iterations, best of 100 runs:
+-------------------------+------+-----------+-----------+
|                    name | size | make (ms) | item (ms) |
+-------------------------+------+-----------+-----------+
|       dict from literal | 0000 |    0.6856 |    9.8825 |
|      tuple from literal | 0000 |    0.6847 |    0.6849 |
|           regular class | 0000 |    0.5362 |    0.6852 |
|       class using slots | 0000 |    0.5363 |    0.6853 |
|  collections namedtuple | 0000 |    0.5354 |    0.6861 |
|  proc. typed NamedTuple | 0000 |    0.6869 |    0.6856 |
|        typed NamedTuple | 0000 |    0.5353 |    0.6838 |
|       regular dataclass | 0000 |    0.5366 |    0.5335 |
| regular dataclass slots | 0000 |    0.6886 |    0.5335 |
|        frozen dataclass | 0000 |    0.6897 |    0.5335 |
|  frozen dataclass slots | 0000 |    0.6899 |    0.5336 |
+-------------------------+------+-----------+-----------+
"""

##############
# Statistics #
##############

print = print if PRINTSTATS else id


def stats(x):
  print(f"{x:.6f}" if isinstance(x, float) else " ".join(f"{_x:.6f}" for _x in x))


ratios_4700U_to_8700k = [
  85.1526 / 77.4804,
  8.2798 / 7.7768,
  393.2533 / 419.9600,
  341.9360 / 370.3981,
  358.9367 / 404.1549,
  358.3562 / 406.0698,
  361.5343 / 402.0191,
  395.0910 / 417.8476,
  345.4267 / 367.6712,
  348.0640 / 377.8337,
  746.4775 / 794.3493,
  677.4594 / 752.0625,
  679.9086 / 760.5823,
  22.5792 / 24.0198,
  20.2340 / 24.3028,
  23.0611 / 22.1962,
  22.6880 / 21.2082,
  20.7772 / 19.9666,
  23.3201 / 19.7323,
  21.2919 / 20.0104,
  24.0210 / 22.4429,
  21.2310 / 21.0758,
  22.3947 / 21.2881,
  23.1717 / 22.7376,
  20.7030 / 21.3258,
  21.2003 / 21.7042,
  68.9289 / 67.4939,
  5.6050 / 4.8811,
  321.4695 / 343.6451,
  292.5633 / 314.6168,
  294.4185 / 324.5867,
  296.6719 / 325.6120,
  294.3780 / 325.4169,
  321.2171 / 344.2590,
  291.5041 / 316.9539,
  634.3641 / 697.5164,
  594.0366 / 645.1800,
  16.5077 / 16.3071,
  15.3206 / 14.7222,
  17.6356 / 19.4977,
  20.0772 / 20.3658,
  17.1632 / 18.5107,
  17.6866 / 18.3033,
  17.1658 / 18.2874,
  18.1571 / 19.5564,
  18.7349 / 20.3715,
  17.6353 / 19.4942,
  20.1110 / 20.6960,
  0.6991 / 0.5567,
  0.5326 / 0.5622,
  0.5349 / 0.5567,
  0.5353 / 0.5614,
  0.5343 / 0.5600,
  0.6858 / 0.5567,
  0.5342 / 0.5567,
  0.5354 / 0.5567,
  0.5357 / 0.5680,
  0.5369 / 0.5567,
  0.5372 / 0.5567,
  13.5548 / 13.4704,
  0.6842 / 0.5600,
  0.6003 / 0.5568,
  0.5346 / 0.7157,
  0.6852 / 0.5600,
  0.5335 / 0.5567,
  0.6834 / 0.5570,
  0.6847 / 0.5567,
  0.5351 / 0.5658,
  0.6849 / 0.5568,
  0.5815 / 0.5567,
]
"""
from these ratios of 4700U's performance to the 8700k's performance, we observe:
- min: 0.746961
- max: 1.255793
- mean: 0.992118
- geometric_mean: 0.986834
- harmonic_mean: 0.981818
- median: 0.958955
- quartiles: 0.922671 0.958955 1.041622
- deciles: 0.901171 0.919671 0.930853 0.941268 0.958955 0.971359 1.020610 1.064552 1.217790
- pstdev: 0.105231
- pvariance: 0.011074
- stdev: 0.105991
- variance: 0.011234
"""
print("Satistics from ratio of 4700U / 8700k performance")
stats(min(ratios_4700U_to_8700k))
stats(max(ratios_4700U_to_8700k))
stats(mean(ratios_4700U_to_8700k))
stats(geometric_mean(ratios_4700U_to_8700k))
stats(harmonic_mean(ratios_4700U_to_8700k))
stats(median(ratios_4700U_to_8700k))
stats(quantiles(ratios_4700U_to_8700k))
stats(quantiles(ratios_4700U_to_8700k, n=10))
stats(pstdev(ratios_4700U_to_8700k))
stats(pvariance(ratios_4700U_to_8700k))
stats(stdev(ratios_4700U_to_8700k))
stats(variance(ratios_4700U_to_8700k))
print("")

ratios_cpython_to_pypy = [
  68.9289 / 0.6991,
  5.6050 / 0.5326,
  321.4695 / 0.5349,
  292.5633 / 0.5353,
  294.4185 / 0.5343,
  296.6719 / 0.6858,
  294.3780 / 0.5342,
  321.2171 / 0.5354,
  291.5041 / 0.5357,
  634.3641 / 0.5369,
  594.0366 / 0.5372,
  16.5077 / 13.5548,
  15.3206 / 0.6842,
  17.6356 / 0.6003,
  20.0772 / 0.5346,
  17.1632 / 0.6852,
  17.6866 / 0.5335,
  17.1658 / 0.6834,
  18.1571 / 0.6847,
  18.7349 / 0.5351,
  17.6353 / 0.6849,
  20.1110 / 0.5815,
  67.4939 / 0.5567,
  4.8811 / 0.5622,
  343.6451 / 0.5567,
  314.6168 / 0.5614,
  324.5867 / 0.5600,
  325.6120 / 0.5567,
  325.4169 / 0.5567,
  344.2590 / 0.5567,
  316.9539 / 0.5680,
  697.5164 / 0.5567,
  645.1800 / 0.5567,
  16.3071 / 13.4704,
  14.7222 / 0.5600,
  19.4977 / 0.5568,
  20.3658 / 0.7157,
  18.5107 / 0.5600,
  18.3033 / 0.5567,
  18.2874 / 0.5570,
  19.5564 / 0.5567,
  20.3715 / 0.5658,
  19.4942 / 0.5568,
  20.6960 / 0.5567,
]
"""
from these ratios of CPython's performance to PyPy's performance, we observe:
- min: 1.210588
- max: 1252.948446
- mean: 306.739916
- geometric_mean: 94.033356
- harmonic_mean: 17.201756
- median: 36.590494
- quartiles: 28.686328 36.590494 574.818000
- deciles: 16.457918 26.289643 32.855084 35.011135 36.590494 432.592447 554.540351 584.896713 862.096938
- pstdev: 367.905574
- pvariance: 135354.511379
- stdev: 372.158959
- variance: 138502.290713
"""
print("Satistics from ratio of CPython / PyPy performance")
stats(min(ratios_cpython_to_pypy))
stats(max(ratios_cpython_to_pypy))
stats(mean(ratios_cpython_to_pypy))
stats(geometric_mean(ratios_cpython_to_pypy))
stats(harmonic_mean(ratios_cpython_to_pypy))
stats(median(ratios_cpython_to_pypy))
stats(quantiles(ratios_cpython_to_pypy))
stats(quantiles(ratios_cpython_to_pypy, n=10))
stats(pstdev(ratios_cpython_to_pypy))
stats(pvariance(ratios_cpython_to_pypy))
stats(stdev(ratios_cpython_to_pypy))
stats(variance(ratios_cpython_to_pypy))
print("")

ratios_4700U_battery_to_4700U_plugged = [
  77.4804 / 129.8398,
  7.7768 / 13.1828,
  419.9600 / 775.1731,
  370.3981 / 461.8491,
  404.1549 / 463.4976,
  406.0698 / 468.3234,
  402.0191 / 449.3548,
  417.8476 / 459.1070,
  367.6712 / 424.1747,
  377.8337 / 452.0948,
  794.3493 / 1191.4967,
  752.0625 / 901.3104,
  760.5823 / 843.3253,
  24.0198 / 46.1178,
  24.3028 / 51.6418,
  22.1962 / 44.0287,
  21.2082 / 26.3737,
  19.9666 / 20.9656,
  19.7323 / 19.5120,
  20.0104 / 19.7859,
  22.4429 / 22.4903,
  21.0758 / 21.3831,
  21.2881 / 28.5871,
  22.7376 / 37.7340,
  21.3258 / 21.8362,
  21.7042 / 23.4631,
  67.4939 / 114.3563,
  4.8811 / 8.4461,
  343.6451 / 640.5819,
  314.6168 / 577.8327,
  324.5867 / 493.8302,
  325.6120 / 348.2305,
  325.4169 / 342.9820,
  344.2590 / 393.6214,
  316.9539 / 370.9233,
  697.5164 / 861.5859,
  645.1800 / 1050.7468,
  16.3071 / 28.1532,
  14.7222 / 26.9623,
  19.4977 / 31.9918,
  20.3658 / 32.3710,
  18.5107 / 33.8196,
  18.3033 / 18.1998,
  18.2874 / 19.0184,
  19.5564 / 21.3467,
  20.3715 / 21.2177,
  19.4942 / 34.6186,
  20.6960 / 29.0954,
  0.5567 / 0.9427,
  0.5622 / 0.9630,
  0.5567 / 0.9559,
  0.5614 / 0.9464,
  0.5600 / 0.9443,
  0.5567 / 0.9441,
  0.5567 / 0.9447,
  0.5567 / 1.6885,
  0.5680 / 1.7018,
  0.5567 / 3.6359,
  0.5567 / 0.9501,
  13.4704 / 25.8458,
  0.5600 / 0.9480,
  0.5568 / 1.2427,
  0.7157 / 0.9418,
  0.5600 / 0.9432,
  0.5567 / 0.9437,
  0.5570 / 1.1097,
  0.5567 / 1.2091,
  0.5658 / 0.9422,
  0.5568 / 2.1674,
  0.5567 / 0.9422,
]
"""
from these ratios of unplugged performance to plugged performance for the 4700U, we observe:
- min: 0.153112
- max: 1.011346
- mean: 0.687106
- geometric_mean: 0.653819
- harmonic_mean: 0.610916
- median: 0.601543
- quartiles: 0.574212 0.601543 0.872624
- deciles: 0.473737 0.544788 0.584442 0.590610 0.601543 0.731331 0.848872 0.908482 0.961419
- pstdev: 0.198292
- pvariance: 0.039320
- stdev: 0.199723
- variance: 0.039889
"""
print("Satistics from ratio of 4700U battery / 4700U plugged performance")
stats(min(ratios_4700U_battery_to_4700U_plugged))
stats(max(ratios_4700U_battery_to_4700U_plugged))
stats(mean(ratios_4700U_battery_to_4700U_plugged))
stats(geometric_mean(ratios_4700U_battery_to_4700U_plugged))
stats(harmonic_mean(ratios_4700U_battery_to_4700U_plugged))
stats(median(ratios_4700U_battery_to_4700U_plugged))
stats(quantiles(ratios_4700U_battery_to_4700U_plugged))
stats(quantiles(ratios_4700U_battery_to_4700U_plugged, n=10))
stats(pstdev(ratios_4700U_battery_to_4700U_plugged))
stats(pvariance(ratios_4700U_battery_to_4700U_plugged))
stats(stdev(ratios_4700U_battery_to_4700U_plugged))
stats(variance(ratios_4700U_battery_to_4700U_plugged))
print("")
