from pyspark.sql.dataframe import DataFrame

from data_lake_machine_learning.configs.evaluators.binary_classification_metrics_config import (
    BinaryClassificationMetricsEvaluatorConfig,
)
from data_lake_machine_learning.operations.evaluators.base_evaluator import (
    BaseEvaluator,
)
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import pyspark.sql.functions as F

from data_lake_machine_learning.common.logger import ml_logger


class BinaryClassificationMetricsEvaluator(BaseEvaluator):
    """Base operation, template for all operations for validation of dataframes"""

    def evaluate(self, df: DataFrame) -> None:
        assert isinstance(self.config, BinaryClassificationMetricsEvaluatorConfig)
        params = self.config.evaluation_params
        df = df.select([F.col(params.label_column), F.col(params.prediction_column)])
        evaluation_metrics = BinaryClassificationMetrics(df.rdd)
        aroc = evaluation_metrics.areaUnderROC
        self._log_evaluator_run(f"Area under ROC is: {aroc}")
