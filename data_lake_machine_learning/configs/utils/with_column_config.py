from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import Any, ClassVar, List, Type, Optional
from data_lake_machine_learning.configs.base import BaseConfig


@dataclass
class WithColumnConfig:

    name: str
    expr: str
    ml_function: Optional[str] = field(default=None)
    Schema: ClassVar[Type[Schema]] = Schema  # noqa


@dataclass
class WithColumnOperationConfig(BaseConfig):
    """
    Operation configuration for reading raw data (csv, delta, parquet) from dataio data source into data frame
    """

    params: List[WithColumnConfig]

    Schema: ClassVar[Type[Schema]] = Schema  # noqa
