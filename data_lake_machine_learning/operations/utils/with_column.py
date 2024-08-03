from importlib import import_module
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F
from data_lake_machine_learning.configs.utils.with_column_config import (
    WithColumnOperationConfig,
)
from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base import BaseOperation


class WithColumnOperation(BaseOperation):

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, WithColumnOperationConfig)
        df = mlc[self.config.data_from_operation]
        for p in self.config.params:
            if p.ml_function:
                ml_func = self._find_ml_function(p.ml_function)
                df = df.withColumn(p.name, ml_func(F.expr(p.expr)))
            else:
                df = df.withColumn(p.name, F.expr(p.expr))
        return df

    def _find_ml_function(self, func_name):
        op_file = import_module("pyspark.ml.functions")
        ml_func = getattr(op_file, func_name)
        return ml_func


