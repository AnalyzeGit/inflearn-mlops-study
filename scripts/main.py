from ml.ml_pipeline import MLManager  
from mlops_config import DATA_DIR

if __name__=="__main__":
    # 데이터 로드 
    data_path = DATA_DIR / "train.csv"
    # 타겟 값 설정정
    target_col = "채무 불이행 여부"

    # 실행 객체 생성 
    ml_instance = MLManager(data_path, target_col)
    ml_instance.predict_with_selected_features()