from abc import abstractmethod
from pyspark.sql.dataframe import DataFrame
from data_lake_machine_learning.configs.base import BaseConfig
from data_lake_machine_learning.operations.base import BaseOperation
from pyspark.sql.session import SparkSession
from data_lake_machine_learning.common.string_helper import (
    snake_case_to_camel_case_with_space,
)
from data_lake_machine_learning.operations.evaluators.base_evaluator import (
    BaseEvaluator,
)
from typing import List
from data_lake_machine_learning.common.logger import ml_logger


class BaseMLOperation(BaseOperation):

    def __init__(
        self,
        config: BaseConfig,
        spark_session: SparkSession,
        evaluators: List[BaseEvaluator] = None,
    ):
        self.evaluators = evaluators
        super().__init__(config=config, spark_session=spark_session)

    @abstractmethod
    def _train(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError(self)

    def _log_evaluaton_run(self) -> None:
        ml_logger.info(
            snake_case_to_camel_case_with_space(self.config.operation_name),
            self.config.name,
        )


