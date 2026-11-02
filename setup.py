from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Package metadata lives in pyproject.toml. This file only defines the Cython extension.
extensions = [
    Extension(
        name="elastic_st.weighted_elastic_net",
        sources=["elastic_st/weighted_elastic_net.pyx"],
        include_dirs=[numpy.get_include()],
        # Wheels are built against NumPy 2.x; this keeps them importable with NumPy 1.23 and newer.
        define_macros=[("NPY_TARGET_VERSION", "NPY_1_23_API_VERSION")],
    ),
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
