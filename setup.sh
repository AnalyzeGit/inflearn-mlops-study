#!/bin/bash

# 1. 가상환경 생성
pyenv virtualenv 3.11.1 mlops_study_env
pyenv activate mlops_study_env

# 2. pip 최신화
pip install --upgrade pip

# 3. 프로젝트 의존성 설치 (editable 설치)
pip install -e .

# (선택) Jupyter에서 venv 인식시키기
pip install ipykernel
python -m ipykernel install --user --name=mlops-study --display-name "Python (mlops-study)"

echo "✅ 가상환경 설치 및 패키지 설치 완료" 