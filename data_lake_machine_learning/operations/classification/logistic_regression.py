from pyspark.sql.dataframe import DataFrame
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from data_lake_machine_learning.configs.classification.logistic_regression_config import (
    LogisticRegressionOperationConfig,
)

from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base_ml import BaseMLOperation


class LogisticRegressionOperation(BaseMLOperation):
    """
    Validates the uniqueness in input dataframe according to column set defined in cofiguration.
    Depending on configuration setting, the dataframe can be de-duplicated.
    """

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, LogisticRegressionOperationConfig)
        df = mlc[self.config.data_from_operation]

        if self.config.retrain:
            df = self._train(df)
        else:
            model = LogisticRegressionModel.load(self.config.model_path)
            df = model.transform(df)
        return df

    def _train(self, df: DataFrame) -> DataFrame:
        params = self.config.params
        log_regression = LogisticRegression(
            featuresCol=params.features_column,
            labelCol=params.label_column,
            predictionCol=params.prediction_column,
            maxIter=params.max_iter,
            regParam=params.regression_param,
            elasticNetParam=params.elastic_net_param,
            fitIntercept=params.fit_intercept,
            threshold=params.threshold,
            probabilityCol=params.probability_column,
            rawPredictionCol=params.raw_prediction_column,
            standardization=params.standardization,
            aggregationDepth=params.aggregation_depth,
            family=params.family,
        )
        training_df, testing_df = df.randomSplit(
            [self.config.training_data, self.config.testing_data]
        )
        model = log_regression.fit(training_df)
        model.write().overwrite().save(self.config.model_path)
        df = model.transform(testing_df)
        self._log_evaluaton_run()
        for e in self.evaluators:
            e.evaluate(df)
        return df
