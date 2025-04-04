from setuptools import setup, find_packages

setup(
    name="compound_trading",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "python-binance",
        "scikit-learn",
        "torch",
        "tqdm"
    ],
)