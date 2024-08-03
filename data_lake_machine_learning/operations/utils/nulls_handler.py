from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F
from data_lake_machine_learning.configs.utils.nulls_handler_config import (
    NullsHandlerOperationConfig,
)
from data_lake_machine_learning.operations.base import BaseOperation
from data_lake_machine_learning.machine_learning_context import MachineLearningContext


class NullsHandlerOperation(BaseOperation):
    """
    Validates the uniqueness in input dataframe according to column set defined in cofiguration.
    Depending on configuration setting, the dataframe can be de-duplicated.
    """

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self._config, NullsHandlerOperationConfig)
        df = mlc[self.config.data_from_operation]
        is_null_statements = []
        for c in df.dtypes:
            dtype = c[1]
            name = c[0]
            for p in self._config.params:
                if c[0] == p.column:
                    expr = p.replace_with
                    is_null_statements.append(
                        F.when(
                            F.col(name).isNull(), F.expr(expr).cast(dtype).alias(name)
                        )
                        .otherwise(F.col(name))
                        .alias(name)
                    )

        return df.select(is_null_statements)
