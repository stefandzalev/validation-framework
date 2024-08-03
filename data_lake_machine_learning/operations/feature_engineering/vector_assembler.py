from pyspark.sql.dataframe import DataFrame
from pyspark.ml.feature import VectorAssembler
from data_lake_machine_learning.configs.feature_engineering.vector_assembler_config import (
    VectorAssemblerOperationConfig,
)
from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base import BaseOperation
import pyspark.sql.functions as F


class VectorAssemblerOperation(BaseOperation):
    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, VectorAssemblerOperationConfig)
        df = mlc[self._config.data_from_operation]
        df_not_null = df.filter(
            " and ".join([f"{c} is not null" for c in self.config.params.input_columns])
        )
        df_null = df.subtract(df_not_null)
        df_null = df_null.withColumn(self.config.params.output_column, F.lit(None))
        vec_assembler = VectorAssembler(
            inputCols=self.config.params.input_columns,
            outputCol=self.config.params.output_column,
            handleInvalid=self.config.params.handle_invalid,
        )
        return vec_assembler.transform(df_not_null).union(df_null)
