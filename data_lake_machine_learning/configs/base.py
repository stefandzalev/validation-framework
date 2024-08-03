from marshmallow_dataclass import dataclass
from typing import ClassVar, Optional, Type
from marshmallow import Schema, pre_load


@dataclass
class BaseConfig:
    """Base config for that is used in all Machine Learning operations"""

    name: str
    """Name of the operation, can be any name describing the process step"""
    data_from_operation: Optional[str]
    """Name of operation this operation should use data from, default is last"""
    category: str
    """Category of Machine Learning algorithm"""
    operation_name: str
    """Name of operation to be used from the category"""

    Schema: ClassVar[Type[Schema]] = Schema  # noqa # type: ignore

    @pre_load
    def fill_defaults(self, conf, **_kwargs):
        conf["data_from_operation"] = conf.get("data_from_operation", "__last__")
        return conf
