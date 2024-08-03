from pyspark.sql.dataframe import DataFrame
from pyspark.ml.clustering import KMeans, KMeansModel
from data_lake_machine_learning.configs.clustering.k_means_config import (
    KMeansOperationConfig,
)


from data_lake_machine_learning.machine_learning_context import MachineLearningContext
from data_lake_machine_learning.operations.base_ml import BaseMLOperation
import pyspark.sql.functions as F
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import Window
from pyspark.ml.functions import vector_to_array


class KMeansOperation(BaseMLOperation):

    def execute(self, mlc: MachineLearningContext) -> DataFrame:
        assert isinstance(self.config, KMeansOperationConfig)
        df = mlc[self.config.data_from_operation]
        df_nulls = df.filter(F.col(self.config.params.features_column).isNull())
        df = df.subtract(df_nulls)
        if self.config.retrain:
            df = self._train(df)
        else:
            model = KMeansModel.load(self.config.model_path)
            df = model.transform(df)
            if self.config.params.center_column is not None:
                df = self._add_centers(model, df)
        df_nulls = df_nulls.withColumn("outlier_factor", F.lit(0.0))
        return df.unionByName(df_nulls, allowMissingColumns=True)

    def _train(self, df: DataFrame) -> DataFrame:
        k = self.config.params.k
        if self.config.params.auto_k:
            k = self._auto_determine_k(df)
        kmeans = KMeans(
            featuresCol=self.config.params.features_column,
            distanceMeasure=self.config.params.distance_measure,
            predictionCol=self.config.params.prediction_column,
            seed=self.config.params.seed,
            k=k,
            maxIter=self.config.params.max_iter,
        )
        model = kmeans.fit(df)
        model.write().overwrite().save(self.config.model_path)
        df = model.transform(df)
        self._log_evaluaton_run()
        for e in self.evaluators:
            e.evaluate(df)
        if self.config.params.center_column is not None:
            df = self._add_centers(model, df)
        return df

    def _add_centers(self, model: KMeans, df: DataFrame) -> DataFrame:
        centers = [c[0] for c in model.clusterCenters()]
        print(centers)
        print(self.config.params.center_column)
        print(self.config.params.prediction_column)

        df = df.withColumn(
            self.config.params.center_column,
            F.array([F.lit(c) for c in centers]).getItem(
                F.col(self.config.params.prediction_column)
            ),
        )
        df = df.withColumn(
            "distance_to_centre",
            F.abs(
                vector_to_array(F.col(self.config.params.features_column))[0]
                - F.col("center")
            ),
        )
        p_wind = Window.partitionBy(self.config.params.prediction_column)
        df = df.withColumn("max_dist", F.max(F.col("distance_to_centre")).over(p_wind))
        df = df.withColumn(
            "outlier_factor", F.col("distance_to_centre") / F.col("max_dist")
        )
        return df

    def _auto_determine_k(self, df: DataFrame) -> int:

        silhouette_score = []
        max_score = 0
        k = 2
        params = self.config.params
        evaluator = ClusteringEvaluator(
            predictionCol=params.prediction_column,
            featuresCol=params.features_column,
            metricName=params.evaluation_metric,
        )
        for i in range(2, 20):
            kmeans = KMeans(
                featuresCol=params.features_column,
                predictionCol=params.prediction_column,
                seed=params.seed,
                k=i,
                maxIter=params.max_iter,
            )
            model = kmeans.fit(df)
            predictions = model.transform(df)
            score = evaluator.evaluate(predictions)
            silhouette_score.append(score)
            if score > max_score:
                max_score = score
                k = i
        return k
