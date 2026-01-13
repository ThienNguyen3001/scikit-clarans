from setuptools import setup, find_packages

setup(
    name='scikit-clarans',
    version='0.1.0',
    description='A scikit-learn compatible implementation of CLARANS clustering algorithm',
    author='User',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy'
    ],
)
