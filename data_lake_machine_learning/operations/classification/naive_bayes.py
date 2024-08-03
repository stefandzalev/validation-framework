from pyspark.sql.dataframe import DataFrame
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from data_lake_machine_learning.configs.classification.naive_bayes_config import (
    NaiveBayesOperationConfig,
)


from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base_ml import BaseMLOperation


class NaiveBayesOperation(BaseMLOperation):
    """
    Validates the uniqueness in input dataframe according to column set defined in cofiguration.
    Depending on configuration setting, the dataframe can be de-duplicated.
    """

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, NaiveBayesOperationConfig)
        df = mlc[self.config.data_from_operation]

        if self.config.retrain:
            df = self._train(df)
        else:
            model = NaiveBayesModel.load(self.config.model_path)
            df = model.transform(df)
        return df

    def _train(self, df: DataFrame) -> DataFrame:
        params = self.config.params
        naive_bayes = NaiveBayes(
            featuresCol=params.features_column,
            labelCol=params.label_column,
            predictionCol=params.prediction_column,
            probabilityCol=params.probability_column,
            rawPredictionCol=params.raw_prediction_column,
            smoothing=params.smoothing,
            modelType=params.model_type,
        )
        training_df, testing_df = df.randomSplit(
            [self.config.training_data, self.config.testing_data]
        )
        model = naive_bayes.fit(training_df)
        model.write().overwrite().save(self.config.model_path)
        df = model.transform(training_df)
        self._log_evaluaton_run()
        for e in self.evaluators:
            e.evaluate(df)
        return df
