from typing import Dict
import yaml
from data_lake_machine_learning.machine_learning_context import ValidationContext
from data_lake_machine_learning.operations.operation_factory import OperationFactory
from pyspark.sql.session import SparkSession
from jinja2 import Template
from data_lake_machine_learning.common.logger import ml_logger
from data_lake_machine_learning.common.string_helper import (
    snake_case_to_camel_case_with_space,
)


class ValidationProcess:
    def __init__(
        self, process_path: str, spark_session: SparkSession, params: Dict = None
    ):
        self.process_path = process_path
        self._spark_session = spark_session
        self.params = params or {}
    def _read_yaml(self) -> Dict:
        with open(self.process_path, "r", encoding="utf-8") as f:
            output = yaml.safe_load(Template(f.read()).render(self.params))
        return output
    def run(self, mlc_initial: ValidationContext = None) -> ValidationContext:
        mlc = mlc_initial or ValidationContext({})
        conf = self._read_yaml()
        ml_logger.info(
            "Starting process %s",
            conf["name"],
        )
        for o in conf["operations"]:
            if o["category"] in ["clustering", "classification", "regression"]:
                op = OperationFactory.create_ml_operation(
                    operation_config=o, spark_session=self._spark_session
                )
            else:
                op = OperationFactory.create_operation(
                    operation_config=o, spark_session=self._spark_session
                )
            ml_logger.info(
                "Running: %s operation from category: %s with name: %s",
                snake_case_to_camel_case_with_space(o["operation_name"]),
                snake_case_to_camel_case_with_space(o["category"]),
                o["name"],
            )
            res_mlc = op.execute(mlc)
            mlc[op.config.name] = res_mlc
            mlc.last = res_mlc
        return mlc


