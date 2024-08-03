from dataclasses import field
from marshmallow import Schema  # noqa
from marshmallow_dataclass import dataclass
from typing import ClassVar, Optional, Type
from data_lake_machine_learning.configs.base import BaseConfig
from data_lake_machine_learning.configs.base_ml import BaseSupervisedConfig


@dataclass
class RandomForestConfig:
    """ """

    features_column: str = field(default="features")
    label_column: str = field(default="label")
    prediction_column: str = field(default="prediction")
    probability_column: str = field(default="probability")
    raw_prediction_column: str = field(default="raw_prediction")
    max_depth: int = field(default=5)
    max_bins: int = field(default=32)
    min_instances_per_node: int = field(default=1)
    min_info_gain: float = field(default=0.0)
    checkpoint_interval: int = field(default=10)
    impurity: str = field(default="gini")
    num_trees: int = field(default=20)
    feature_subset_strategy: str = field(default="auto")
    sub_sampling_rate: float = field(default=1.0)
    leaf_column: str = field(default="")
    bootstrap: bool = field(default=True)
    weight_column: str = field(default=None)
    seed: int = field(default=None)

    Schema: ClassVar[Type[Schema]] = Schema  # type: ignore # noqa


@dataclass
class RandomForestOperationConfig(BaseSupervisedConfig):
    """
    Configuration for uniqueness validation operation and deduplication
    """

    params: Optional[RandomForestConfig] = field(default=RandomForestConfig())

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore
