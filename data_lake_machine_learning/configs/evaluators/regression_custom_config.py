from typing import ClassVar, Type
from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from data_lake_machine_learning.configs.evaluators.base_evaluator import (
    BaseEvaluatorConfig,
)


@dataclass
class RegressionCustomConfig:
    """ """

    prediction_column: str = field(default="prediction")
    label_column: str = field(default="label")
    metric_name: str = field(default="rmse")
    through_origin: bool = field(default=False)

    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class RegressionCustomEvaluatorConfig(BaseEvaluatorConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    evaluation_params: RegressionCustomConfig = field(default=RegressionCustomConfig())

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
