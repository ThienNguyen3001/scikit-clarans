"""Helper script to build Cython extensions for scikit-clarans."""
import os
import sys
import shutil
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

def build():
    # Identify compiler
    compiler_flag = []
    if sys.platform == "win32":
        # Check if Strawberry GCC is available and MSVC is not
        has_cl = shutil.which("cl.exe") is not None
        has_gcc = shutil.which("gcc.exe") is not None
        if not has_cl and has_gcc:
            compiler_flag = ["--compiler=mingw32"]

    extensions = [
        Extension(
            "clarans._core",
            sources=["clarans/_core.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"] if ("--compiler=mingw32" in compiler_flag or sys.platform != "win32") else ["/O2"],
        )
    ]

    sys_argv_bak = sys.argv
    sys.argv = ["setup.py", "build_ext", "--inplace"] + compiler_flag

    try:
        setup(
            ext_modules=cythonize(
                extensions,
                language_level=3,
                compiler_directives={
                    "boundscheck": False,
                    "wraparound": False,
                    "cdivision": True,
                    "initializedcheck": False,
                },
            ),
        )
        print("\n[INFO] Cython build successful.")
    finally:
        sys.argv = sys_argv_bak

if __name__ == "__main__":
    build()
