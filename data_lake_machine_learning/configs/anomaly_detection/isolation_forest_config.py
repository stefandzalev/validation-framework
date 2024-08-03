from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, Optional, Type
from data_lake_machine_learning.configs.base import BaseConfig
from data_lake_machine_learning.configs.base_ml import BaseMLConfig


@dataclass
class IsolationForestConfig:
    """ """

    path: str = field(default="")
    num_trees: int = field(default=100)
    max_samples: int = field(default=500)
    contamination: float = field(default=0.0)
    approx_quantile_relative_error: float = field(default=0.0)
    max_features: float = field(default=1.0)
    bootstrap: bool = field(default=False)
    max_depth: int = field(default=10)
    features_column: str = field(default="features")
    prediction_column: str = field(default="prediction")
    anomaly_score: str = field(default="score")
    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class IsolationForestOperationConfig(BaseMLConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    params: IsolationForestConfig = field(default=IsolationForestConfig())

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
