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


class ml_manager:
    def __init__(self) -> None:
        self.df = pd.read_csv(r'../data/train.csv')

    def _select_features_by_logistic_lasso(
            self,
            target_col: str, 
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
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
            ]
        )

        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=random_state))
        ])

        clf.fit(X, y)
        encoded_feature_names = clf.named_steps['preprocessor'].get_feature_names_out()
        coefs = clf.named_steps['classifier'].coef_[0]
        selected_features = [feature for coef, feature in zip(coefs, encoded_feature_names) if coef != 0]

        return selected_features

    def predict_with_selected_features(
            self, 
            target_col: str, 
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
        selected_features = self._select_features_by_logistic_lasso(target_col)

        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
            ]
        )

        X_processed = preprocessor.fit_transform(X)
        all_feature_names = preprocessor.get_feature_names_out()
        
        selected_idx = [i for i, name in enumerate(all_feature_names) if name in selected_features]
        X_selected = X_processed[:, selected_idx]

        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=random_state)
        clf = LogisticRegression(solver='liblinear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

        return y_pred, clf
