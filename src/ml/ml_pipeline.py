# basic
import pandas as pd
import numpy as np

# type
from typing import List, Tuple

# [ml] data-preprocessing 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# [ml] modeling & evaluation 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# [ml] pipeline
from sklearn.pipeline import Pipeline

# package
from dataline.data_handler import DataHandler

class MLManager:
    def __init__(self, data_path: str, target_col: str) -> None:
        date_hanlder = DataHandler(data_path)
        self.X, self.y = date_hanlder.split_features_target(target_col)
        self.categorical_cols, self.numeric_cols = date_hanlder.identify_categorical_numeric_columns(self.X)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.categorical_cols)
            ]
        )

    def _select_features_by_logistic_lasso(
            self,
            C: float = 1.0, 
            random_state: int = 42
            ) -> List[str]:
        """
        Lasso 로지스틱 회귀 기반 변수 선택 함수
        
        Parameters:
            target_col (str): 타겟 변수 컬럼명
            C (float): 규제 강도 (작을수록 규제가 강해짐)
            random_state (int): 랜덤 시드

        Returns:
            selected_features (list): 선택된 변수명 리스트
        """
        clf = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=random_state))
        ])

        clf.fit(self.X, self.y)
        encoded_feature_names = clf.named_steps['preprocessor'].get_feature_names_out()
        coefs = clf.named_steps['classifier'].coef_[0]
        selected_features = [feature for coef, feature in zip(coefs, encoded_feature_names) if coef != 0]

        return selected_features

    def predict_with_selected_features(
            self, 
            test_size: float = 0.2, 
            random_state: int = 42
            ) -> Tuple[np.ndarray, LogisticRegression]:
        """
        선택된 변수만 사용해 로지스틱 회귀로 예측 수행

        Parameters:
            target_col (str): 타겟 컬럼명
            test_size (float): 테스트 데이터 비율
            random_state (int): 랜덤 시드

        Returns:
            y_pred (np.ndarray): 예측 결과 (0 또는 1)
            clf (LogisticRegression): 학습된 모델
            
        """
        selected_features = self._select_features_by_logistic_lasso()

        X_processed = self.preprocessor.fit_transform(self.X)
        all_feature_names = self.preprocessor.get_feature_names_out()
        
        selected_idx = [i for i, name in enumerate(all_feature_names) if name in selected_features]
        X_selected = X_processed[:, selected_idx]

        X_train, X_test, y_train, y_test = train_test_split(X_selected, self.y, test_size=test_size, random_state=random_state)
        clf = LogisticRegression(solver='liblinear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

        return y_pred, clf
