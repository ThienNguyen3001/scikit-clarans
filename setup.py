#! /usr/bin/env python
import codecs
import os
import re

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def find_version() -> str:
    version_file = os.path.join(here, "clarans", "__init__.py")
    with codecs.open(version_file, encoding="utf-8") as f:
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', f.read())
        if match:
            return match.group(1)
    raise RuntimeError("Unable to find version string in clarans/__init__.py.")


setup(
    name="scikit-clarans",
    version=find_version(),
    description="A scikit-learn compatible implementation of CLARANS clustering algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThienNguyen3001/scikit-clarans",
    author="Ngọc Thiện Nguyễn",
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
    package_data={"clarans": ["py.typed"]},
    include_package_data=True,
    install_requires=["numpy", "scikit-learn", "scipy"],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "flake8",
            "pandas",
            "sphinx>=5.0",
            "sphinx-rtd-theme",
            "sphinx-copybutton",
            "sphinx-autodoc-typehints",
        ],
        "test": ["pytest", "pytest-cov", "flake8", "pandas"],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme",
            "sphinx-copybutton",
            "sphinx-autodoc-typehints",
        ],
    },
    python_requires=">=3.9",
    test_suite="clarans.tests",
)
