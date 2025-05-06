
import sys
import os

# 현재 파일 위치 기준으로 src 폴더 경로를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.ml_pipeline import MLManager  

data_path = r'../data/train.csv'
target_col = "채무 불이행 여부"

ml_instance = MLManager(data_path, target_col)
ml_instance.predict_with_selected_features()