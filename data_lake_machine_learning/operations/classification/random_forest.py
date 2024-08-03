from pyspark.sql.dataframe import DataFrame
from pyspark.ml.classification import (
    RandomForestClassifier,
    RandomForestClassificationModel,
)
from data_lake_machine_learning.configs.classification.random_forest_config import (
    RandomForestOperationConfig,
)


from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base_ml import BaseMLOperation


class RandomForestOperation(BaseMLOperation):
    """
    Validates the uniqueness in input dataframe according to column set defined in cofiguration.
    Depending on configuration setting, the dataframe can be de-duplicated.
    """

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, RandomForestOperationConfig)
        df = mlc[self.config.data_from_operation]

        if self.config.retrain:
            df = self._train(df)
        else:
            model = RandomForestClassificationModel.load(self.config.model_path)
            df = model.transform(df)
        return df

    def _train(self, df: DataFrame) -> DataFrame:
        params = self.config.params
        random_forest = RandomForestClassifier(
            featuresCol=params.features_column,
            labelCol=params.label_column,
            predictionCol=params.prediction_column,
            probabilityCol=params.probability_column,
            rawPredictionCol=params.raw_prediction_column,
            maxDepth=params.max_depth,
            maxBins=params.max_bins,
            minInstancesPerNode=params.min_instances_per_node,
            minInfoGain=params.min_info_gain,
            checkpointInterval=params.checkpoint_interval,
            impurity=params.impurity,
            numTrees=params.num_trees,
            featureSubsetStrategy=params.feature_subset_strategy,
            subsamplingRate=params.sub_sampling_rate,
            leafCol=params.leaf_column,
            bootstrap=params.bootstrap,
        )
        training_df, testing_df = df.randomSplit(
            [self.config.training_data, self.config.testing_data]
        )
        model = random_forest.fit(training_df)
        model.write().overwrite().save(self.config.model_path)
        df = model.transform(training_df)
        self._log_evaluaton_run()
        for e in self.evaluators:
            e.evaluate(df)
        return df
