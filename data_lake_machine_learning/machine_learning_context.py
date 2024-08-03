from typing import Dict
from pyspark.sql.dataframe import DataFrame


class MachineLearningContext:
    """
    Stores results from all operations defined in yaml.
    Every operation step results in dataframe, which is stored in attribute of this class.
    """

    def __init__(self, res: Dict[str, DataFrame]):
        self._res = res

    @property
    def last(self) -> DataFrame:
        """Get lastly added operation result and process runs"""
        return self._res["__last__"]

    @last.setter
    def last(self, df: DataFrame) -> None:
        """Set the last result from lastly executed operation"""
        self._res["__last__"] = df

    def __getitem__(self, key: str) -> DataFrame:
        """Get particular result of operation defined in the process, depending on key (operation name)."""
        return self._res[key]

    def __setitem__(self, key: str, df: DataFrame) -> None:
        """Set particular result of operation defined in the process, depending on key (operation name)."""
        self._res[key] = df
