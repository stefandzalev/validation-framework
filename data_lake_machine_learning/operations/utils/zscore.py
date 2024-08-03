from pyspark.sql.dataframe import DataFrame

from data_lake_machine_learning.configs.utils.zscore_config import ZscoreOperationConfig
from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base import BaseOperation
import pyspark.sql.functions as F


class ZscoreOperation(BaseOperation):
    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, ZscoreOperationConfig)
        df = mlc[self.config.data_from_operation]
        params = self.config.params
        df_non_null = df.filter(F.col(params.column).isNotNull())
        df_null = df.subtract(df_non_null)
        stats = df_non_null.select(
            F.mean(F.col(params.column)).alias("mean"),
            F.stddev(F.col(params.column)).alias("stddev"),
        ).collect()[0]
        df_zscore = df_non_null.withColumn(
            f"{params.column}_z_score",
            (F.col(params.column) - stats["mean"]) / stats["stddev"],
        )
        df_valid = df_zscore.filter(
            f"abs({params.column}_z_score) <= {params.factor}"
        ).drop(f"{params.column}_z_score")
        df_non_valid = df_non_null.subtract(df_valid)
        df_valid = df_valid.withColumn("outlier", F.lit(False))
        df_null = df_null.withColumn("outlier", F.lit(False))
        df_non_valid = df_non_valid.withColumn("outlier", F.lit(True))
        return df_valid.union(df_non_valid).union(df_null)
