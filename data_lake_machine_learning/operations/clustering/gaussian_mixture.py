from pyspark.sql.dataframe import DataFrame
from pyspark.ml.clustering import GaussianMixture, GaussianMixtureModel
from data_lake_machine_learning.configs.clustering.gaussian_mixture_config import (
    GaussianMixtureOperationConfig,
)
from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base_ml import BaseMLOperation


class GaussianMixtureOperation(BaseMLOperation):
    """
    Validates the uniqueness in input dataframe according to column set defined in cofiguration.
    Depending on configuration setting, the dataframe can be de-duplicated.
    """

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, GaussianMixtureOperationConfig)
        df = mlc[self.config.data_from_operation]
        if self.config.retrain:
            df = self._train(df)
        else:
            model = GaussianMixtureModel.load(self.config.model_path)
            df = model.transform(df)
        return df

    def _train(self, df: DataFrame) -> DataFrame:
        params = self.config.params
        gaussian_mixture = GaussianMixture(
            featuresCol=params.features_column,
            predictionCol=params.prediction_column,
            k=params.k,
            probabilityCol=params.probability_column,
            tol=params.tol,
            maxIter=params.max_iter,
            seed=params.seed,
            aggregationDepth=params.aggregation_depth,
        )
        model = gaussian_mixture.fit(df)
        model.write().overwrite().save(self.config.model_path)
        df = model.transform(df)
        self._log_evaluaton_run()
        for e in self.evaluators:
            e.evaluate(df)
        return df
