from pyspark.sql.dataframe import DataFrame

from data_lake_machine_learning.configs.utils.inter_quartile_range_assessment_config import (
    InterQuartileRangeAssessmentOperationConfig,
)
from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base import BaseOperation
import pyspark.sql.functions as F


class InterQuartileRangeAssessmentOperation(BaseOperation):
    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, InterQuartileRangeAssessmentOperationConfig)
        df = mlc[self.config.data_from_operation]
        params = self.config.params
        df_non_null = df.filter(F.col(params.column).isNotNull())
        df_null = df.subtract(df_non_null)
        quantiles = df_non_null.approxQuantile(params.column, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1
        lower_bound = q1 - params.factor * iqr
        upper_bound = q3 + params.factor * iqr
        df_valid = df_non_null.filter(
            (F.col(params.column) >= lower_bound)
            & (F.col(params.column) <= upper_bound)
        )
        df_non_valid = df_non_null.subtract(df_valid)
        df_valid = df_valid.withColumn("outlier", F.lit(False))
        df_null = df_null.withColumn("outlier", F.lit(False))
        df_non_valid = df_non_valid.withColumn("outlier", F.lit(True))

        return df_valid.union(df_non_valid).union(df_null)


