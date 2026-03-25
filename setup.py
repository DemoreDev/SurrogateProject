from setuptools import setup, find_packages

setup(
    name="best_pipeline_search", 
    version="0.1.0",
    description="Sistema de busca para pipelines de " \
    "classificação multi-label usando modelos surrogados",
    author="Leonardo Demore",
    
    packages=find_packages(),
    
    install_requires=[
        "pandas",
        "numpy",
        "xgboost",
        "joblib",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],
    
    python_requires=">=3.8",
)