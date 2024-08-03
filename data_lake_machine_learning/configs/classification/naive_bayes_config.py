from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, Optional, Type
from data_lake_machine_learning.configs.base import BaseConfig
from data_lake_machine_learning.configs.base_ml import BaseSupervisedConfig


@dataclass
class NaiveBayesConfig:
    """ """

    features_column: str = field(default="features")
    label_column: str = field(default="label")
    prediction_column: str = field(default="prediction")
    probability_column: str = field(default="probability")
    raw_prediction_column: str = field(default="raw_prediction")
    smoothing: float = field(default=1.0)
    model_type: str = field(default="multinomial")

    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class NaiveBayesOperationConfig(BaseSupervisedConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    params: Optional[NaiveBayesConfig] = field(default=NaiveBayesConfig())

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
