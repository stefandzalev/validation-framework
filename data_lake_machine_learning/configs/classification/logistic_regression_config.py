from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, Optional, Type
from data_lake_machine_learning.configs.base import BaseConfig
from data_lake_machine_learning.configs.base_ml import BaseSupervisedConfig


@dataclass
class LogisticRegressionConfig:
    """ """

    features_column: str = field(default="features")
    label_column: str = field(default="label")
    prediction_column: str = field(default="prediction")
    max_iter: int = field(default=100)
    regression_param: float = field(default=0.0)
    elastic_net_param: float = field(default=0.0)
    fit_intercept: bool = field(default=True)
    threshold: float = field(default=0.5)
    probability_column: str = field(default="probability")
    raw_prediction_column: str = field(default="raw_prediction")
    standardization: bool = field(default=True)
    aggregation_depth: int = field(default=2)
    family: str = field(default="auto")

    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class LogisticRegressionOperationConfig(BaseSupervisedConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    params: LogisticRegressionConfig = field(default=LogisticRegressionConfig())

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
