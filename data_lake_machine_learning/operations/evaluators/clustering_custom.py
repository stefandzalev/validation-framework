from pyspark.sql.dataframe import DataFrame
from data_lake_machine_learning.configs.evaluators.clustering_custom_config import (
    ClusteringCustomEvaluatorConfig,
)
from data_lake_machine_learning.operations.evaluators.base_evaluator import (
    BaseEvaluator,
)
from pyspark.ml.evaluation import ClusteringEvaluator


class ClusteringCustomEvaluator(BaseEvaluator):
    """Base operation, template for all operations for validation of dataframes"""

    def evaluate(self, df: DataFrame) -> None:
        assert isinstance(self.config, ClusteringCustomEvaluatorConfig)
        params = self.config.evaluation_params
        evaluator = ClusteringEvaluator(
            predictionCol=params.prediction_column,
            featuresCol=params.features_column,
            distanceMeasure=params.distance_measure,
            metricName=params.evaluation_metric,
        )
        wcss = evaluator.evaluate(df)
        self._log_evaluator_run(f"Within-Cluster Sum of Squares (WCSS):{wcss}")
