import importlib
from typing import Any, Dict, Tuple, Union
from data_lake_machine_learning.configs.base import BaseConfig
from data_lake_machine_learning.configs.base_ml import BaseMLConfig
from data_lake_machine_learning.operations.base import BaseOperation
from pyspark.sql.session import SparkSession

from data_lake_machine_learning.operations.base_ml import BaseMLOperation
from data_lake_machine_learning.operations.evaluators.base_evaluator import (
    BaseEvaluator,
)


class OperationFactory:
    """Factory for operations, logic for creating particular operation objects and their configurations"""

    @classmethod
    def _find_class_by_name(cls, class_name: str) -> str:
        """
        Creates class name from the operation type definition.
        Idea here is to dynamically be able to import specific operation class and it's config class.
        """
        name_parts = class_name.split("_")
        return "".join([p.capitalize() for p in name_parts])

    @classmethod
    def _prep_class_and_config(
        cls,
        sub_module: str,
        operation_file_name: str,
        config_suffix: str,
        operation_suffix: str,
        operation_config: Dict,
        operations_module: str = "data_lake_machine_learning.operations",
        config_module: str = "data_lake_machine_learning.configs",
    ) -> Tuple[BaseOperation, BaseConfig]:
        # operation_suffix da napram da e config default
        # da sredam operations module config module mozi treba kako vo konstruktor da se
        operation_config_file_name = f"{operation_file_name}_{config_suffix}"

        # Get name in "class name format" using the name in the type from the yaml
        operation_class_name = cls._find_class_by_name(
            f"{operation_file_name}_{operation_suffix}"
        )
        operation_config_class_name = cls._find_class_by_name(
            operation_config_file_name.replace(
                config_suffix, f"_{operation_suffix}_{config_suffix}"
            )
        )
        # Import operation and operation config classes
        operation_file = importlib.import_module(
            f"{operations_module}.{sub_module}.{operation_file_name}"
        )
        operation_config_file = importlib.import_module(
            f"{config_module}.{sub_module}.{operation_config_file_name}"
        )

        operation_class = getattr(operation_file, operation_class_name)
        operation_config_class_name = getattr(
            operation_config_file, operation_config_class_name
        )

        operation_config_class = operation_config_class_name.Schema().load(
            operation_config
        )
        return operation_class, operation_config_class

    @classmethod
    def create_operation(
        cls,
        operation_config: Dict,
        spark_session: SparkSession,
        operations_module: str = "data_lake_machine_learning.operations",
        config_module: str = "data_lake_machine_learning.configs",
    ) -> BaseOperation:
        """
        Returns instance of particular operation, using it's intended config.
        """
        operation_class, operation_config_class = cls._prep_class_and_config(
            operation_config["category"],
            operation_config["operation_name"],
            "config",
            "operation",
            operation_config,
        )
        return operation_class(
            operation_config_class,
            spark_session,
        )

    @classmethod
    def create_ml_operation(
        cls,
        operation_config: Dict,
        spark_session: SparkSession,
        operations_module: str = "data_lake_machine_learning.operations",
        config_module: str = "data_lake_machine_learning.configs",
    ) -> BaseMLOperation:
        """
        Returns instance of particular operation, using it's intended config.
        """

        operation_class, operation_config_class = cls._prep_class_and_config(
            operation_config["category"],
            operation_config["operation_name"],
            "config",
            "operation",
            operation_config,
        )
        evaluators = []
        for e in operation_config["evaluators_config"]:
            evaluators.append(cls._create_evaluator(e))

        return operation_class(
            operation_config_class,
            spark_session,
            evaluators,
        )

    @classmethod
    def _create_evaluator(
        cls,
        evaluator_config: Dict,
        operations_module: str = "data_lake_machine_learning.operations",
        config_module: str = "data_lake_machine_learning.configs",
    ) -> BaseEvaluator:
        evaluator_class, evaluator_config_class = cls._prep_class_and_config(
            "evaluators",
            evaluator_config["evaluator_name"],
            "config",
            "evaluator",
            evaluator_config,
        )
        return evaluator_class(evaluator_config_class)
