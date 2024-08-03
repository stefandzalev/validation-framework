from pyspark.sql.dataframe import DataFrame
from pyspark.ml.feature import StandardScaler, StandardScalerModel
from data_lake_machine_learning.configs.feature_engineering.standard_scaler_config import (
    StandardScalerOperationConfig,
)

from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base_ml import BaseMLOperation
import pyspark.sql.functions as F


class StandardScalerOperation(BaseMLOperation):
    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, StandardScalerOperationConfig)
        df = mlc[self.config.data_from_operation]
        df_null = df.filter(F.col(self.config.params.input_column).isNull())

        df_not_null = df.filter(F.col(self.config.params.input_column).isNotNull())
        df = df_not_null
        if self.config.retrain:
            df = self._train(df)
        else:
            model = StandardScalerModel.load(self.config.model_path)
            df = model.transform(df)
        return df.unionByName(df_null, allowMissingColumns=True)

    def _train(self, df: DataFrame) -> DataFrame:
        standardscaler = StandardScaler(
            withMean=self.config.params.with_mean,
            withStd=self.config.params.with_std,
            inputCol=self.config.params.input_column,
            outputCol=self.config.params.output_column,
        )
        model = standardscaler.fit(df)
        model.write().overwrite().save(self.config.model_path)
        df = model.transform(df)
        return df
