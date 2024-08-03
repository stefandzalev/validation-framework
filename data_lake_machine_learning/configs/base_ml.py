from dataclasses import field
from marshmallow_dataclass import dataclass
from typing import Any, ClassVar, Dict, Optional, Type, List
from marshmallow import Schema, pre_load

from data_lake_machine_learning.configs.base import BaseConfig


@dataclass
class BaseMLModelConfig(BaseConfig):
    """Base config for that is used in all Machine Learning operations"""

    model_path: str
    """"""
    retrain: bool
    """"""
    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore

    @pre_load
    def fill_defaults(self, conf, **_kwargs):
        conf["retrain"] = conf.get("retrain", False)
        return conf


@dataclass
class BaseMLConfig(BaseMLModelConfig):
    """Base config for that is used in all Machine Learning operations"""

    evaluators_config: Optional[List[Dict[str, Any]]]
    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore


@dataclass
class BaseSupervisedConfig(BaseMLConfig):
    """Base config for that is used in all Machine Learning operations"""

    training_data: Optional[float]
    testing_data: Optional[float]
    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore

    @pre_load
    def fill_defaults(self, conf, **_kwargs):
        conf["training_data"] = conf.get("training_data", 0.8)
        conf["testing_data"] = conf.get("testing_data", 0.2)
        if conf.get("training_data", 0) + conf.get("testing_data", 0) != 1:
            raise ValueError(
                f"Training and testing data proportions do not include full dataset."
            )
        return conf
