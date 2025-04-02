from setuptools import setup, find_packages

# Read README.md as long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlopsstudy",  # 패키지 이름
    version="0.1",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},  # src 폴더 기준으로 패키지 찾음
    install_requires=[
        'scikit-learn>=1.0',
        'pandas>=1.3',
        'numpy>=1.21',
    ],
    author="김준형", 
    author_email="wnsgud4553@gmail.com",
    description="A Machine Learning pipeline project focused on MLOps",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/AnalyzeGit/inflearn-mlops-study.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11.1',
)
