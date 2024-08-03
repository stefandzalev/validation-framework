from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, List, Optional, Type

from data_lake_machine_learning.configs.base import BaseConfig


@dataclass
class NullsHandlerConfig:
    """
    Find nulls in specified collumns, if specified replace them with other values
    """

    column: str
    """Column to be checked if it is null"""
    replace_with: Optional[str]
    """Value or expression to be used for replacing the null value"""
    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class NullsHandlerOperationConfig(BaseConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    params: List[NullsHandlerConfig]

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
