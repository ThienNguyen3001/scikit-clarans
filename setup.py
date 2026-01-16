#! /usr/bin/env python
import codecs
import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="scikit-clarans",
    version="0.1.0",
    description="A scikit-learn compatible implementation of CLARANS clustering algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThienNguyen3001/scikit-clarans",
    author="ThienNguyen3001",
    author_email="thiennguyen03001@gmail.com", 
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    keywords="clustering sklearn scikit-learn clarans k-medoids",
    packages=find_packages(),
    install_requires=["numpy", "scikit-learn", "scipy"],
    python_requires=">=3.8",
    test_suite="clarans.tests",
)
