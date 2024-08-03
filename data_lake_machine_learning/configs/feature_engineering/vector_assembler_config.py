from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, List, Optional, Type
from marshmallow.validate import OneOf
from data_lake_machine_learning.configs.base import BaseConfig


@dataclass
class VectorAssemblerConfig:
    """
    Find nulls in specified collumns, if specified replace them with other values
    """

    input_columns: Optional[List[str]]
    """Input columns for feature vector"""
    output_column: str
    """Output column name from vectorization of inputCols"""
    handle_invalid: str = field(
        default="error",
        metadata=dict(validate=OneOf(["skip", "keep", "error"])),
    )
    """
    How to handle invalid data (NULL and NaN values). 
    Options are 'skip' (filter out rows with invalid data), 
    'error' (throw an error), 
    or 'keep' (return relevant number of NaN in the output)
    """
    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class VectorAssemblerOperationConfig(BaseConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    params: VectorAssemblerConfig

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
