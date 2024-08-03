from pyspark.sql.dataframe import DataFrame
from data_lake_machine_learning.configs.evaluators.regression_custom_config import (
    RegressionCustomEvaluatorConfig,
)
from data_lake_machine_learning.operations.evaluators.base_evaluator import (
    BaseEvaluator,
)
from pyspark.ml.evaluation import RegressionEvaluator


class RegressionCustomEvaluator(BaseEvaluator):
    """Base operation, template for all operations for validation of dataframes"""

    def evaluate(self, df: DataFrame) -> None:
        assert isinstance(self.config, RegressionCustomEvaluatorConfig)
        params = self.config.evaluation_params
        evaluator = RegressionEvaluator(
            predictionCol=params.prediction_column,
            labelCol=params.label_column,
            metricName=params.metric_name,
            throughOrigin=params.through_origin,
        )
        lr_eval = evaluator.evaluate(df)
        self._log_evaluator_run(f"Metric: {params.metric_name} = {lr_eval}")
