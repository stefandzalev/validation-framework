from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import Any, ClassVar, List, Type
from data_lake_machine_learning.configs.base import BaseConfig


@dataclass
class SelectConfig:

    columns: List[str]

    Schema: ClassVar[Type[Schema]] = Schema  # noqa


@dataclass
class SelectOperationConfig(BaseConfig):
    """
    Operation configuration for reading raw data (csv, delta, parquet) from dataio data source into data frame
    """

    params: SelectConfig

    Schema: ClassVar[Type[Schema]] = Schema  # noqa
