from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Type
from data_lake_machine_learning.configs.base import BaseConfig
from marshmallow.validate import OneOf


@dataclass
class ReadConfig:

    path: str
    options: Dict[str, Any] = field(default_factory=dict)
    format: str = field(
        default="parquet",
        metadata=dict(validate=OneOf(["csv", "json", "parquet", "xml"])),
    )

    Schema: ClassVar[Type[Schema]] = Schema  # noqa


@dataclass
class ReadOperationConfig(BaseConfig):
    """
    Operation configuration for reading raw data (csv, delta, parquet) from dataio data source into data frame
    """

    params: ReadConfig

    Schema: ClassVar[Type[Schema]] = Schema  # noqa
