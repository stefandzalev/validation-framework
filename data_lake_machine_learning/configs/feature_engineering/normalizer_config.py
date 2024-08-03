from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, Type
from data_lake_machine_learning.configs.base import BaseConfig


@dataclass
class NormalizerConfig:
    """ """

    column_name: str


@dataclass
class NormalizerOperationConfig(BaseConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    params: NormalizerConfig

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
