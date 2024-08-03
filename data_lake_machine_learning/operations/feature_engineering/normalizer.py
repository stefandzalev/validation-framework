from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F
from data_lake_machine_learning.configs.feature_engineering.normalizer_config import (
    NormalizerOperationConfig,
)
from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base import BaseOperation


class NormalizerOperation(BaseOperation):
    """
    Validates the uniqueness in input dataframe according to column set defined in cofiguration.
    Depending on configuration setting, the dataframe can be de-duplicated.
    """

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, NormalizerOperationConfig)
        df = mlc[self.config.data_from_operation]
        orig_cols = df.columns
        params = self.config.params
        max_value = df.agg(
            F.max(F.col(params.column_name)).alias(f"{params.column_name}_max")
        )
        min_value = df.agg(
            F.min(F.col(params.column_name)).alias(f"{params.column_name}_min")
        )
        df = df.crossJoin(min_value).crossJoin(max_value)
        # df.select(
        #     F.col(f"{params.column_name}_min"), F.col(f"{params.column_name}_max")
        # ).show()
        df = df.withColumn(
            f"{params.column_name}_normalized",
            (F.col(params.column_name) - F.col(f"{params.column_name}_min"))
            / (F.col(f"{params.column_name}_max") - F.col(f"{params.column_name}_min")),
        )
        orig_cols.append(f"{params.column_name}_normalized")
        return df.select(*orig_cols)
