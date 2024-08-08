"""
`@attr_docs` lets `Enum` classes get attribute `__doc__` support.

Copyright 2024 Alex Blandin
"""

import ast
import inspect
import itertools


class AttrDocProperty(property):
  """
  Property attribute.

    fget
      function to be used for getting an attribute value
    fset
      function to be used for setting an attribute value
    fdel
      function to be used for del'ing an attribute
    doc
      docstring

  Typical use is to define a managed attribute x:
  ```
  class C(object):
      def getx(self): return self._x
      def setx(self, value): self._x = value
      def delx(self): del self._x
      x = property(getx, setx, delx, "I'm the 'x' property.")
  ```
  Decorators make defining new properties or modifying existing ones easy:
  ```
  class C(object):
      @property
      def x(self):
          "I am the 'x' property."
          return self._x
      @x.setter
      def x(self, value):
          self._x = value
      @x.deleter
      def x(self):
          del self._x
  ```
  """

  def __str__(self) -> str:
    return self.__doc__ if self.__doc__ is not None else super().__str__()

  def __repr__(self) -> str:
    return repr(self.__doc__) if self.__doc__ is not None else super().__repr__()


def attr_docs(cls):
  """
  Fixes `__doc__` on Enum class attributes, so they can be documented too, in accordance with PEP257.

  The machinery is a `property` that replaces `cls.__doc__`, wrapped so `cls.__doc__.__str__` works.

  This, of course, means anything that inspects `cls.__doc__` too closely will fail, but c'est la vie.

  ```
  @attr_docs
  class Words(StrEnum):
    "A typical enum class"

    hello = auto()
    "A greeting"

    bye = goodbye = auto()
    "The opposite of a greeting"

  assert Words.hello.__doc__ == "A greeting"
  ```
  """

  clast = next(e for e in ast.walk(ast.parse(inspect.getsource(cls))) if isinstance(e, ast.ClassDef))
  __attr_docs__ = {clast.name: cls.__doc__}

  for expr, nexpr in itertools.pairwise(clast.body):
    if (
      isinstance(expr, ast.AnnAssign | ast.Assign)
      and isinstance(nexpr, ast.Expr)
      and isinstance(doc_string := nexpr.value.value, str)
    ):
      for target in (expr.target,) if isinstance(expr, ast.AnnAssign) else expr.targets:
        __attr_docs__[ast.unparse(target)] = doc_string.strip()

  if __attr_docs__:
    cls.__attr_docs__ = __attr_docs__

    @AttrDocProperty
    def attr_doc(self=cls) -> str | None:
      return (
        self.__class__.__attr_docs__.get(self)
        or (self.__class__.__attr_docs__.get(self.name) if hasattr(self, "name") else None)
        or (self.__class__.__attr_docs__.get(self.__name__) if hasattr(self, "__name__") else None)
        or self.__class__.__attr_docs__.get(self.__class__.__name__)
        or None
      )

    attr_doc.__doc__ = cls.__doc__
    cls.__doc__ = attr_doc

  return cls


import enum


@attr_docs
class Words(enum.StrEnum):
  "A typical enum class"

  hello = enum.auto()
  "A greeting"

  bye = goodbye = enum.auto()
  "The opposite of a greeting"


@attr_docs
class Swords(enum.Enum):
  "A collection of ..."

  short: int = 42
  "More of a dagger, really"

  great = 9001
  r"""
  Now, this, this is a sword
  ```
                />
  (           //-------------------------------------------------------(
  (*)OXOXOXOXO(*>                  --------                             \
  (           \\---------------------------------------------------------)
                \>
  ```
  """


if __name__ == "__main__":
  # ruff: noqa: T201
  print(f"{Words.hello.__doc__ = :}")
  print(f"{Words.bye.__doc__ = :}")
  print(f"{Words.goodbye.__doc__ = :}")
  print(f"{Words.__doc__ = :}")

  print(f"{Swords.short.__doc__ = :}")
  print(f"{Swords.great.__doc__ = :}")
  print(f"{Swords.__doc__ = :}")
