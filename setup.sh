#!/bin/bash

echo "✅ 가상환경 활성화"

# 이전 가상환경이 있다면 비활성화
deactivate 2>/dev/null

# 가상환경 생성 (폴더명: .venv)
python3 -m venv ./inflearn_mlops_env

# 가상환경 활성화
source ./inflearn_mlops_env/bin/activate 

echo "👉 현재 python 경로: $(which python)"

if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ 가상환경이 활성화되지 않았습니다. 스크립트를 종료합니다."
    exit  1
fi

# pip 최신화
pip install --upgrade pip

# 프로젝트 의존성 설치 (editable 모드)
pip install -e .
 
# 실행
echo "✅ 가상환경 설치 및 패키지 설치 완료"
echo "✅ 스크립트 실행"
python ./scripts/main.py