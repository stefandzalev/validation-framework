from marshmallow_dataclass import dataclass
from typing import ClassVar, Type
from marshmallow import Schema  # noqa


@dataclass
class BaseEvaluatorConfig:
    """Base config for that is used in all Machine Learning operations"""

    evaluator_name: str
    """Name of the operation, can be any name describing the process step"""

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
