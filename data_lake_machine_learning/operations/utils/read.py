from pyspark.sql.dataframe import DataFrame
from data_lake_machine_learning.configs.utils.read_config import ReadOperationConfig

from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base import BaseOperation


class ReadOperation(BaseOperation):
    """
    Validates the uniqueness in input dataframe according to column set defined in cofiguration.
    Depending on configuration setting, the dataframe can be de-duplicated.
    """

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, ReadOperationConfig)
        params = self.config.params
        df_reader = self._spark_session.read.format(params.format)
        if params.options:
            df_reader = df_reader.options(**params.options)
        return df_reader.load(params.path)
