from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, Type

from data_lake_machine_learning.configs.evaluators.base_evaluator import (
    BaseEvaluatorConfig,
)


@dataclass
class ClusteringEvaluatorConfig:
    """ """

    features_column: str = field(default="features")
    distance_measure: str = field(default="squaredEuclidean")
    prediction_column: str = field(default="prediction")
    evaluation_metric: str = field(default="silhouette")

    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class ClusteringCustomEvaluatorConfig(BaseEvaluatorConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    evaluation_params: ClusteringEvaluatorConfig = field(
        default=ClusteringEvaluatorConfig()
    )

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
