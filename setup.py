# Setup file used for `pip install -e`
from setuptools import setup, find_packages

setup(
    name = 'DeepLearningChallenge',
    version = '1.0.0',
    packages = find_packages(),
    install_requires = [
        'ipykernel~=6.29.5',
        'matplotlib~=3.10.3',
        'pandas~=2.2.3',
        'requests~=2.32.3',
        'scikit-learn~=1.6.1',
        'tensorflow~=2.16.2',
    ]
)