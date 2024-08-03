from pyspark.sql.dataframe import DataFrame


from data_lake_machine_learning.configs.evaluators.multiclass_classification_custom_config import (
    MulticlassClassificationCustomEvaluatorConfig,
)
from data_lake_machine_learning.operations.evaluators.base_evaluator import (
    BaseEvaluator,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


class MulticlassClassificationCustomEvaluator(BaseEvaluator):
    """Base operation, template for all operations for validation of dataframes"""

    def evaluate(self, df: DataFrame) -> None:
        assert isinstance(self.config, MulticlassClassificationCustomEvaluatorConfig)
        params = self.config.evaluation_params
        evaluator = MulticlassClassificationEvaluator(
            predictionCol=params.prediction_column,
            labelCol=params.label_column,
            metricName=params.metric_name,
            metricLabel=params.metric_label,
            beta=params.beta,
            probabilityCol=params.probability_column,
            eps=params.eps,
        )
        accuracy = evaluator.evaluate(df)
        self._log_evaluator_run(f"Accuracy is: {accuracy}")
