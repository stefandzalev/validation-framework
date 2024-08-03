from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import Any, ClassVar, List, Type
from data_lake_machine_learning.configs.base import BaseConfig


@dataclass
class FilterConfig:

    conditions: List[str]

    Schema: ClassVar[Type[Schema]] = Schema  # noqa


@dataclass
class FilterOperationConfig(BaseConfig):
    """
    Operation configuration for reading raw data (csv, delta, parquet) from dataio data source into data frame
    """

    params: FilterConfig

    Schema: ClassVar[Type[Schema]] = Schema  # noqa
