"""
POD testing.

Copyright 2022 Alex Blandin

██████╗ ██╗      █████╗ ██╗███╗   ██╗
██╔══██╗██║     ██╔══██╗██║████╗  ██║
██████╔╝██║     ███████║██║██╔██╗ ██║
██╔═══╝ ██║     ██╔══██║██║██║╚██╗██║
██║     ███████╗██║  ██║██║██║ ╚████║
╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝.

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

- attrs, and what to use when you can't use attrs (hint, it's @dataclass)
"""

# TODO(alex): large refurb into something actually usable and not a dirty hack of a test

# currently not identifying machines properly, so here's the Windows versions to machine/CPU mappings
# 10.0.19045 is i7-8700k (look, I didn't realise I hadn't updated from 19041 the first time, have rerun with that now)
# 10.0.22621 is 7980HS (windows 11 shows as windows 10 by major version, ain't that "funny")

import contextlib
import sys
import timeit
from datetime import datetime
from pathlib import Path
from sys import getsizeof

# import cattrs
from .pod_formats import (
  DataClass,
  DataSlots,
  DataSlotsAuto,
  FrozenData,
  FrozenDataSlots,
  FrozenDataSlotsAuto,
  NTuple,
  ProcTypedNTuple,
  Regular,
  Slots,
  Struct,
  StructSubclassable,
  TypedNTuple,
)

MILLI = 10**3
MICRO = 10**6
NANO = 10**9
UNIT = {MILLI: "ms", MICRO: "μs", NANO: "ns"}

###########
## Dials ##
###########

N_ITERATIONS, N_RUNS = 10**6, 1000  # how long a run is, and how many runs to pick the best average/total from
TIMESCALE = NANO  # set


################
## Formatting ##
################


start_time = datetime.now()  # noqa: DTZ005
version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
output = Path(__file__).parent / "pods"
output.mkdir(exist_ok=True)
output = output / f"pod_{sys.platform}_{sys.implementation.name}_{version}_{start_time:%Y-%m-%d-%H-%M-%S}.txt"
output.touch()
buffer = output.open(mode="+a", encoding="utf8", newline="\n")


def print2(*args, **kwargs) -> None:  # noqa: ANN002, ANN003 # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
  "We print to both the terminal and this pod's buffer."
  print(*args, **kwargs)  # noqa: T201 # pyright: ignore[reportUnknownArgumentType]
  print(*args, **kwargs, file=buffer)  # pyright: ignore[reportUnknownArgumentType]


print2(f"POD run commencing {start_time:%Y-%m-%d-%H-%M-%S}")
print("This pod is running on ", end="")  # noqa: T201
if sys.platform == "win32" and sys.getwindowsversion().platform_version:
  major, minor, build = sys.getwindowsversion().platform_version
  print2(f"Windows {major}.{minor} build {build} at {sys.executable}")
else:
  print2(f"{sys.platform} at {sys.executable}")
print2(" ".join(sys.version.splitlines()).replace("  ", " "))
print2()

print("POD results for ", end="")  # noqa: T201
print2(f"{N_ITERATIONS} iterations, averaged, best of {N_RUNS} runs:")
sep = f"+-{'':->23}-+-{'':->4}-+-{'':->11}-+-{'':->11}-+{''}"  # extra {''} is to appease syntax highlighting
print2(sep)
print2(f"| {'name':>23} | size | create ({UNIT[TIMESCALE]}) | access ({UNIT[TIMESCALE]}) |")
print2(sep)


def time(code: str, iterations: int = N_ITERATIONS, runs: int = N_RUNS, setup: str = "") -> float:
  """Time how long something takes, result is min recorded time for an entire run in seconds."""
  return min(timeit.repeat(code, setup=setup, number=iterations, repeat=runs, globals=globals()))


try:
  _ = getsizeof("I want the overhead, not the footprint")
except TypeError:

  def getsizeof(obj: object, default: int = -1) -> int:  # noqa: ARG001
    "Dunder because pypy complains."
    return -1


def row(name: str, new: str, access: str) -> None:
  """Another row in the table."""
  with contextlib.suppress(Exception):
    to_new = TIMESCALE * time(new) / N_ITERATIONS
    to_access = TIMESCALE * time("var" + access, setup="var=" + new) / N_ITERATIONS
    print2(
      f"| {name:>23} | {getsizeof(eval(new)):04} | {to_new:>11.4f} | {to_access:>11.4f} |"  # noqa: S307  # pyright: ignore [reportAny]
    )


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

###########
## Tests ##
###########

row("dict from literal", "{'amount': 1.0, 'receiver': 'me', 'date': '2022-01-01', 'sender': 'you'}", "['receiver']")
row("tuple from literal", "(1.0, 'me', '2022-01-01', 'you')", "[1]")
row("regular class", "Regular(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("class using slots", "Slots(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("collections namedtuple", "NTuple(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row(
  "proc. typed NamedTuple", "ProcTypedNTuple(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver"
)
row("typed NamedTuple", "TypedNTuple(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("regular dataclass", "DataClass(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("regular dataclass slots", "DataSlots(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("autogen dataclass slots", "DataSlotsAuto(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row("frozen dataclass", "FrozenData(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row(
  "frozen dataclass slots", "FrozenDataSlots(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver"
)
row(
  "autogen frozen dc slots",
  "FrozenDataSlotsAuto(amount=1.0, receiver='me', date='2022-01-01', sender='you')",
  ".receiver",
)
row("handier dataclass slots", "Struct(amount=1.0, receiver='me', date='2022-01-01', sender='you')", ".receiver")
row(
  "even handier dataclasss",
  "StructSubclassable(amount=1.0, receiver='me', date='2022-01-01', sender='you')",
  ".receiver",
)

finish_time = datetime.now()  # noqa: DTZ005
print2(sep)
print2()
print2(f"POD run completed {finish_time:%Y-%m-%d-%H-%M-%S}")
print()  # noqa: T201

buffer.close()
