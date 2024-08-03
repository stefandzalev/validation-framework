from abc import ABC, abstractmethod
from typing import Optional
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.session import SparkSession
from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.configs.base import BaseConfig


class BaseOperation(ABC):
    def __init__(self, config: BaseConfig, spark_session: SparkSession) -> None:
        self._config = config
        self._spark_session = spark_session

    @abstractmethod
    def execute(self, mlc: MachineLearningContext) -> Optional[DataFrame]:
        raise NotImplementedError(self)

    @property
    def config(self) -> BaseConfig:
        return self._config


