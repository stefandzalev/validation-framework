from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, List, Optional, Type
from marshmallow.validate import OneOf
from data_lake_machine_learning.configs.base import BaseConfig
from data_lake_machine_learning.configs.base_ml import BaseMLModelConfig


@dataclass
class StandardScalerConfig:
    """ """

    input_column: str = field(default=None)

    output_column: str = field(default=None)

    with_mean: bool = field(default=False)

    with_std: bool = field(default=True)

    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class StandardScalerOperationConfig(BaseMLModelConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    params: StandardScalerConfig

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
