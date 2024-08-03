from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, Optional, Type
from data_lake_machine_learning.configs.base import BaseConfig
from data_lake_machine_learning.configs.base_ml import BaseMLModelConfig


@dataclass
class MinMaxScalerConfig:
    """ """

    input_column: str

    output_column: str

    min: float = field(default=0.0)

    max: float = field(default=1.0)

    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class MinMaxScalerOperationConfig(BaseMLModelConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    params: MinMaxScalerConfig

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
