from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, Type

from data_lake_machine_learning.configs.evaluators.base_evaluator import (
    BaseEvaluatorConfig,
)


@dataclass
class BinaryClassificationMetricsConfig:
    """ """

    label_column: str = field(default="label")
    prediction_column: str = field(default="prediction")

    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class BinaryClassificationMetricsEvaluatorConfig(BaseEvaluatorConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    evaluation_params: BinaryClassificationMetricsConfig = field(
        default=BinaryClassificationMetricsConfig()
    )

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
