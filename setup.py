from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="weighted_elastic_net",
        sources=["elastic_st\\weighted_elastic_net.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name="elastic_st",
    version="0.1.0",
    author="Thomas Gust",
    author_email="thomasgust@seattleacademy.org",
    description="A description of your package",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/your_project",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        'scipy',
        'scikit-learn',
        'matplotlib',
        'networkx'
    ],
    setup_requires=[
        "cython",
        "numpy",
    ],
)