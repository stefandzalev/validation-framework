from pyspark.sql.dataframe import DataFrame
from pyspark.ml.regression import LinearRegression, LinearRegressionModel


from data_lake_machine_learning.configs.regression.linear_regression_config import (
    LinearRegressionOperationConfig,
)
from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base_ml import BaseMLOperation
from data_lake_machine_learning.common.logger import ml_logger


class LinearRegressionOperation(BaseMLOperation):
    """
    Validates the uniqueness in input dataframe according to column set defined in cofiguration.
    Depending on configuration setting, the dataframe can be de-duplicated.
    """

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, LinearRegressionOperationConfig)
        df = mlc[self.config.data_from_operation]
        if self.config.retrain:
            df = self._train(df)
        else:
            model = LinearRegressionModel.load(self.config.model_path)
            df = model.transform(df)
        return df

    def _train(self, df: DataFrame) -> DataFrame:
        params = self.config.params
        linear_regression = LinearRegression(
            featuresCol=params.features_column,
            labelCol=params.label_column,
            predictionCol=params.prediction_column,
            maxIter=params.max_iter,
            regParam=params.regression_param,
            elasticNetParam=params.elastic_net_param,
            tol=params.tol,
            fitIntercept=params.fit_intercept,
            standardization=params.standardization,
            solver=params.solver,
            aggregationDepth=params.aggregation_depth,
            epsilon=params.epsilon,
        )
        training_df, testing_df = df.randomSplit(
            [self.config.training_data, self.config.testing_data]
        )
        model = linear_regression.fit(training_df)
        model.write().overwrite().save(self.config.model_path)
        df = model.transform(testing_df)
        ml_logger.info(
            "Linear regression operation with name %s has: \n Coefficients: %s \n Intercept: %s",
            self.config.name,
            model.coefficients,
            model.intercept,
        )
        self._log_evaluaton_run()
        for e in self.evaluators:
            e.evaluate(df)
        return df
