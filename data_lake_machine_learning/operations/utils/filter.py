from pyspark.sql.dataframe import DataFrame
from data_lake_machine_learning.configs.utils.filter_config import FilterOperationConfig

from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base import BaseOperation


class FilterOperation(BaseOperation):
    
    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, FilterOperationConfig)
        df = mlc[self.config.data_from_operation]
        params = self.config.params
        return df.filter(" AND ".join(params.conditions))
