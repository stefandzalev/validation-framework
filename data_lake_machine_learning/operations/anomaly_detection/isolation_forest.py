from pyspark.sql.dataframe import DataFrame
from data_lake_machine_learning.configs.anomaly_detection.isolation_forest_config import (
    IsolationForestOperationConfig,
)
from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base_ml import BaseMLOperation
from pyspark_iforest.ml.iforest import IForestModel, IForest
import pyspark.sql.functions as F
import time

from pyspark.mllib.evaluation import BinaryClassificationMetrics


class IsolationForestOperation(BaseMLOperation):
    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, IsolationForestOperationConfig)
        df = mlc[self.config.data_from_operation]
        df_nulls = df.filter(F.col(self.config.params.features_column).isNull())
        df = df.subtract(df_nulls)
        if self.config.retrain:
            df = self._train(df)
        else:
            model = IForestModel.load(self.config.model_path)
            df = model.transform(df)
        return df

    def _train(self, df: DataFrame) -> DataFrame:
        assert isinstance(self.config, IsolationForestOperationConfig)
        params = self.config.params
        num_trees = self._auto_determine_num_trees(df)
        i_forest = IForest(
            featuresCol=params.features_column,
            predictionCol=params.prediction_column,
            scoreCol=params.anomaly_score,
            numTrees=num_trees,
            maxSamples=params.max_samples,
            maxFeatures=params.max_features,
            maxDepth=params.max_depth,
            contamination=params.contamination,
            bootstrap=params.bootstrap,
            approxQuantileRelativeError=params.approx_quantile_relative_error,
        )
        model = i_forest.fit(df)
        model.write().overwrite().save(self.config.model_path)
        df = model.transform(df)
        return df

    def _auto_determine_num_trees(self, df: DataFrame) -> int:
        roc_score = []
        max_score = 0
        num_trees = {
            0: 100,
            1: 200,
            2: 300,
            3: 400,
            4: 500,
        }
        training_df = df
        params = self.config.params
        k = 0
        for i in range(5):
            i_forest = IForest(
                featuresCol=params.features_column,
                predictionCol=params.prediction_column,
                scoreCol=params.anomaly_score,
                numTrees=num_trees[i],
                maxSamples=params.max_samples,
                maxFeatures=params.max_features,
                maxDepth=params.max_depth,
                contamination=params.contamination,
                bootstrap=params.bootstrap,
                approxQuantileRelativeError=params.approx_quantile_relative_error,
            )
            model = i_forest.fit(training_df)
            res_df = model.transform(training_df).select(
                [F.col("anomalyScore"), F.col(params.prediction_column)]
            )
            evaluator = BinaryClassificationMetrics(res_df.rdd)
            roc_score.append(evaluator.areaUnderROC)
            if roc_score[i] > max_score:
                max_score = roc_score[i]
                k = i
        return num_trees[k]



