# basic
import pandas as pd
import numpy as np

# typing
from typing import Tuple, List

class DataHandler:
    def __init__(self, date_path: str) -> None:
        self.df = pd.read_csv(date_path)

    def split_features_target(self, target_col: str) -> Tuple[pd.DataFrame, pd.Series]: 
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        return (X, y) 

    def identify_categorical_numeric_columns(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numeric_cols =  X.select_dtypes(include=[np.number]).columns.tolist()

        return categorical_cols, numeric_cols