from abc import ABC, abstractmethod
from pyspark.sql.dataframe import DataFrame
from data_lake_machine_learning.configs.evaluators.base_evaluator import (
    BaseEvaluatorConfig,
)
from data_lake_machine_learning.common.logger import ml_logger
from data_lake_machine_learning.common.string_helper import (
    snake_case_to_camel_case_with_space,
)


class BaseEvaluator(ABC):
    """Base operation, template for all operations for validation of dataframes"""

    def __init__(self, config: BaseEvaluatorConfig) -> None:
        self._config = config

    @abstractmethod
    def evaluate(self, df: DataFrame) -> None:
        """Logic for specific implementation in each sub-class for different types of validations"""
        raise NotImplementedError(self)

    @property
    def config(self) -> BaseEvaluatorConfig:
        """Returns the configuration for the particular operation"""
        return self._config

    def _log_evaluator_run(self, msg: str) -> None:
        eval_name = snake_case_to_camel_case_with_space(self._config.evaluator_name)
        ml_logger.info("Running %s evaluator", eval_name)
        ml_logger.info("Result from %s evaluator is: %s", eval_name, msg)
