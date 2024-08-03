from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, List, Optional, Type
from dataclasses import field
from marshmallow.validate import OneOf
from data_lake_machine_learning.configs.base import BaseConfig


@dataclass
class DuplicatesHandlerConfig:
    """
    Validate and de-duplicate data, based on columns in configuration
    """

    check_columns: List[str]
    """ Columns to be used for checking (correcting) duplicates in data """
    deduplicate_based_on: List[str]
    """ Columns to be used to order duplicated data based on check_columns """
    deduplication_order: Optional[str] = field(
        default="asc",
        metadata=dict(validate=OneOf(["asc", "desc"])),
    )
    """ Describes the ordering of deduplication, can be asc or desc """
    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class DuplicatesHandlerOperationConfig(BaseConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    params: DuplicatesHandlerConfig

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
