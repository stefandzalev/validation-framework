from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from dataclasses import field
from typing import ClassVar, Type
from data_lake_machine_learning.configs.base import BaseConfig


@dataclass
class ZscoreConfig:

    column: str
    factor: float = field(default=1.5)

    Schema: ClassVar[Type[Schema]] = Schema  # noqa


@dataclass
class ZscoreOperationConfig(BaseConfig):
    """
    Operation configuration for reading raw data (csv, delta, parquet) from dataio data source into data frame
    """

    params: ZscoreConfig

    Schema: ClassVar[Type[Schema]] = Schema  # noqa
