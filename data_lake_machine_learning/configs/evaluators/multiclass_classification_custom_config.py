from typing import ClassVar, Type
from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from data_lake_machine_learning.configs.evaluators.base_evaluator import (
    BaseEvaluatorConfig,
)


@dataclass
class MulticlassClassificationConfig:
    """ """

    prediction_column: str = field(default="prediction")
    probability_column: str = field(default="probability")
    label_column: str = field(default="label")
    metric_label: float = field(default=0.0)
    beta: float = field(default=1.0)
    eps: float = field(default=1e-15)
    metric_name: str = field(default="f1")

    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class MulticlassClassificationCustomEvaluatorConfig(BaseEvaluatorConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    evaluation_params: MulticlassClassificationConfig = field(
        default=MulticlassClassificationConfig()
    )

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
