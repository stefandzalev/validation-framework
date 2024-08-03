from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Type
from data_lake_machine_learning.configs.base import BaseConfig
from marshmallow.validate import OneOf


@dataclass
class WriteConfig:

    path: str
    options: Dict[str, Any] = field(default_factory=dict)
    format: str = field(
        default="parquet",
        metadata=dict(validate=OneOf(["csv", "json", "parquet", "xml"])),
    )
    mode: str = field(
        default="append",
        metadata=dict(validate=OneOf(["overwrite", "append"])),
    )

    Schema: ClassVar[Type[Schema]] = Schema  # noqa


@dataclass
class WriteOperationConfig(BaseConfig):
    """
    Operation configuration for reading raw data (csv, delta, parquet) from dataio data source into data frame
    """

    params: WriteConfig

    Schema: ClassVar[Type[Schema]] = Schema  # noqa
