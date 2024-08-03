from pyspark.sql.dataframe import DataFrame
from data_lake_machine_learning.configs.utils.select_config import SelectOperationConfig

from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base import BaseOperation


class SelectOperation(BaseOperation):
    """
    Validates the uniqueness in input dataframe according to column set defined in cofiguration.
    Depending on configuration setting, the dataframe can be de-duplicated.
    """

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, SelectOperationConfig)
        df = mlc[self.config.data_from_operation]
        params = self.config.params
        return df.select(*params.columns)
