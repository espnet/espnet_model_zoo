#!/usr/bin/env python3
import os
from setuptools import find_packages
from setuptools import setup


requirements = {
    "install": [
        "pandas",
        "requests",
        "tqdm",
        "numpy",
        "espnet",
        "huggingface_hub",
        "filelock",
    ],
    "setup": ["pytest-runner"],
    "test": [
        "pytest>=3.3.0",
        "pytest-pythonpath>=0.7.3",
        "pytest-cov>=2.7.1",
        "hacking>=1.1.0",
        "mock>=2.0.0",
        "pycodestyle",
        "flake8>=3.7.8",
        "black",
    ],
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {
    k: v for k, v in requirements.items() if k not in ["install", "setup"]
}

dirname = os.path.dirname(__file__)
setup(
    name="espnet_model_zoo",
    version="0.1.0",
    url="http://github.com/espnet/espnet_model_zoo",
    description="ESPnet Model Zoo",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache Software License",
    packages=find_packages(include=["espnet_model_zoo*"]),
    package_data={"espnet_model_zoo": ["table.csv"]},
    entry_points={
        "console_scripts": [
            "espnet_model_zoo_upload = espnet_model_zoo.zenodo_upload:main",
            "espnet_model_zoo_download = espnet_model_zoo.downloader:cmd_download",
            "espnet_model_zoo_query = espnet_model_zoo.downloader:cmd_query",
        ],
    },
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    python_requires=">=3.6.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
