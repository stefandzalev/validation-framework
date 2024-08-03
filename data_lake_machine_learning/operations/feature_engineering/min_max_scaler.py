from pyspark.sql.dataframe import DataFrame
from pyspark.ml.feature import MinMaxScaler
from data_lake_machine_learning.configs.feature_engineering.min_max_scaler_config import (
    MinMaxScalerOperationConfig,
)

from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base import BaseOperation


class MinMaxScalerOperation(BaseOperation):
    """
    Validates the uniqueness in input dataframe according to column set defined in cofiguration.
    Depending on configuration setting, the dataframe can be de-duplicated.
    """

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, MinMaxScalerOperationConfig)
        df = mlc[self.config.data_from_operation]
        params = self.config.params
        mm_scaler = MinMaxScaler(
            inputCol=params.input_column,
            outputCol=params.output_column,
            max=params.max,
            min=params.min,
        )
        return mm_scaler.fit(df).transform(df)
