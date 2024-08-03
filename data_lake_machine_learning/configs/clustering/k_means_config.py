from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, Optional, Type
from data_lake_machine_learning.configs.base import BaseConfig
from data_lake_machine_learning.configs.base_ml import BaseMLConfig


@dataclass
class KMeansConfig:
    """ """

    seed: int = field(default=None)
    features_column: str = field(default="features")
    distance_measure: str = field(default="euclidean")
    prediction_column: str = field(default="prediction")
    evaluation_metric: str = field(default="silhouette")
    k: Optional[int] = field(default=2)
    auto_k: Optional[bool] = field(default=False)
    max_iter: int = field(default=500)
    center_column: str = field(default="center")

    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class KMeansOperationConfig(BaseMLConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    params: KMeansConfig = field(default=KMeansConfig())

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
