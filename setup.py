"""
Setup script for wood_identification package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="wood_identification",
    version="0.1.0",
    author="schwinger42",
    description="Wood identification using Raman spectroscopy with ML/DL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/schwinger42/wood_identification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "h5py>=3.1.0",
        "joblib>=1.0.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
        "tensorboard>=2.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    keywords="raman spectroscopy wood identification machine-learning deep-learning",
    project_urls={
        "Bug Reports": "https://github.com/schwinger42/wood_identification/issues",
        "Source": "https://github.com/schwinger42/wood_identification",
    },
)
