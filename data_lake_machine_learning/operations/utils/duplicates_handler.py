from pyspark.sql.dataframe import DataFrame
from data_lake_machine_learning.configs.utils.duplicates_handler_config import (
    DuplicatesHandlerOperationConfig,
)
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from data_lake_machine_learning.operations.base import BaseOperation
from data_lake_machine_learning.machine_learning_context import MachineLearningContext


class DuplicatesHandlerOperation(BaseOperation):
    """
    Validates the uniqueness in input dataframe according to column set defined in cofiguration.
    Depending on configuration setting, the dataframe can be de-duplicated.
    """

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self._config, DuplicatesHandlerOperationConfig)
        df = mlc[self.config.data_from_operation]
        params = self._config.params
        partition_by_cols = params.check_columns
        order_by_cols = params.deduplicate_based_on
        if params.deduplication_order == "desc":
            order_by_cols = [F.desc(c) for c in order_by_cols]
        rn_window = Window.partitionBy(*partition_by_cols).orderBy(*order_by_cols)
        df = df.withColumn("rn", F.row_number().over(rn_window))
        return df.filter(F.col("rn") == 1)
