from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TypeAlias

JSONScalar: TypeAlias = None | bool | int | float | str
LogicalValue: TypeAlias = (
    JSONScalar | Decimal | list["LogicalValue"] | dict[str, "LogicalValue"]
)


@dataclass(frozen=True, slots=True)
class Fact:
    """An unencoded observation owned by exactly one registry producer."""

    key: str
    logical_value: LogicalValue

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise TypeError("fact key must be a non-empty string")
