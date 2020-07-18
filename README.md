# ESPnet Model Zoo

[![PyPI version](https://badge.fury.io/py/espnet_model_zoo.svg)](https://badge.fury.io/py/espnet_model_zoo)
[![Python Versions](https://img.shields.io/pypi/pyversions/espnet_model_zoo.svg)](https://pypi.org/project/espnet_model_zoo/)
[![Downloads](https://pepy.tech/badge/espnet_model_zoo)](https://pepy.tech/project/espnet_model_zoo)
[![GitHub license](https://img.shields.io/github/license/espnet/espnet_model_zoo.svg)](https://github.com/espnet/espnet_model_zoo)
![CI](https://github.com/espnet/espnet_model_zoo/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/espnet/espnet_model_zoo/branch/master/graph/badge.svg)](https://codecov.io/gh/espnet/espnet_model_zoo)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Install

```
pip install espnet_model_zoo
```

## Usage

```python
>>> from espnet_model_zoo.downloader import ModelDownloader
>>> d = ModelDownloader("~/.cache/espnet")  # Specify cachedir
>>> d = ModelDownloader()  # <module_dir> is used as cachedir by default
```

To download and unpack a model file,
you need to input the model name which you want.

```python
>>> d.get_model("user_name/model_name")
{"asr/config.yaml": <config path>, "asr/pretrain.pth": <model path>, ...}
```

You can also get a model with specifying some attributes.

```python
>>> d.get_model(task="asr", corpus="wsj")
```

If multiple models are matched with the condition, latest model is selected.
You can also specify the model with "version" option.

```python
>>> d.get_model(task="asr", corpus="wsj", version=-1)  # Get the latest model
>>> d.get_model(task="asr", corpus="wsj", version=-2)  # Get previous model
```

You can view uploaded model names from our Zenodo community, https://zenodo.org/communities/espnet/, 
or using `get_model_names()`,

```python
>>> d.get_model_names()
[...]
```

You can also show them with specifying some attributes.

```python
>>> d.get_model_names(task="asr")
[...]
```

## Register your model

1. Upload your model to Zenodo: See https://github.com/espnet/espnet for detail.
1. Create a Pull Request to modify [table.csv](espnet_model_zoo/table.csv)
1. Increment the version number of [setup.py](setup.py)
