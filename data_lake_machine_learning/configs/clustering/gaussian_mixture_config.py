from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, Optional, Type
from data_lake_machine_learning.configs.base import BaseConfig
from data_lake_machine_learning.configs.base_ml import BaseMLConfig


@dataclass
class GaussianMixtureConfig:
    """ """

    seed: int = field(default=None)
    weightCol: str = field(default=None)
    features_column: str = field(default="features")
    prediction_column: str = field(default="prediction")
    probability_column: str = field(default="probability")
    k: int = field(default=2)
    max_iter: int = field(default=20)
    tol: float = field(default=0.01)
    aggregation_depth: int = field(default=2)

    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class GaussianMixtureOperationConfig(BaseMLConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    params: Optional[GaussianMixtureConfig] = field(default=GaussianMixtureConfig())

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
