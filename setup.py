#! /usr/bin/env python
import codecs
import os
import re
import shutil
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

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


# ---------------------------------------------------------------------------
# Cython Extension Configuration (Scikit-Learn Standard)
# ---------------------------------------------------------------------------
is_mingw = False
if sys.platform == "win32":
    has_cl = shutil.which("cl.exe") is not None
    has_gcc = shutil.which("gcc.exe") is not None
    if not has_cl and has_gcc:
        is_mingw = True

extra_compile_args = ["-O3"] if (is_mingw or sys.platform != "win32") else ["/O2"]


class BuildExtAutoCompiler(_build_ext):
    """Automatically configure MinGW compiler on Windows if MSVC is not available."""

    def initialize_options(self):
        super().initialize_options()
        if sys.platform == "win32" and is_mingw:
            self.compiler = "mingw32"


cmdclass = {"build_ext": BuildExtAutoCompiler}
ext_modules = []

try:
    import numpy as np
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [
            Extension(
                "clarans._core",
                sources=["clarans/_core.pyx"],
                include_dirs=[np.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                extra_compile_args=extra_compile_args,
            )
        ],
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
            "nonecheck": False,
        },
    )
except ImportError:
    if os.path.exists(os.path.join(here, "clarans", "_core.c")):
        try:
            import numpy as np

            ext_modules = [
                Extension(
                    "clarans._core",
                    sources=["clarans/_core.c"],
                    include_dirs=[np.get_include()],
                    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                    extra_compile_args=extra_compile_args,
                )
            ]
        except ImportError:
            pass


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
    package_data={"clarans": ["py.typed", "*.pyx", "*.pxd"]},
    include_package_data=True,
    install_requires=["numpy", "scikit-learn", "scipy"],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
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
            "cython",
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
)
