from pyspark.sql.dataframe import DataFrame
from data_lake_machine_learning.configs.utils.write_config import WriteOperationConfig

from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base import BaseOperation


class WriteOperation(BaseOperation):
    """
    Validates the uniqueness in input dataframe according to column set defined in cofiguration.
    Depending on configuration setting, the dataframe can be de-duplicated.
    """

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, WriteOperationConfig)
        df = mlc[self.config.data_from_operation]
        params = self.config.params
        df_writer = df.write.format(params.format).mode(params.mode)
        if params.options:
            df_writer = df_writer.options(**params.options)
        df_writer.save(params.path)
        return df
