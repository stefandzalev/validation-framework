from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, Type
from data_lake_machine_learning.configs.base_ml import BaseSupervisedConfig

@dataclass
class LinearRegressionConfig:
    """ """

    features_column: str = field(default="features")
    label_column: str = field(default="label")
    prediction_column: str = field(default="prediction")
    max_iter: int = field(default=100)
    regression_param: float = field(default=0.0)
    elastic_net_param: float = field(default=0.0)
    tol: float = field(default=1e-6)
    fit_intercept: bool = field(default=True)
    standardization: bool = field(default=True)
    solver: str = field(default="auto")
    aggregation_depth: int = field(default=2)
    epsilon: float = field(default=1.35)

    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class LinearRegressionOperationConfig(BaseSupervisedConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    params: LinearRegressionConfig = field(default=LinearRegressionConfig())

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
