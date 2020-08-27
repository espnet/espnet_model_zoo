# ESPnet Model Zoo

[![PyPI version](https://badge.fury.io/py/espnet-model-zoo.svg)](https://badge.fury.io/py/espnet-model-zoo)
[![Python Versions](https://img.shields.io/pypi/pyversions/espnet_model_zoo.svg)](https://pypi.org/project/espnet_model_zoo/)
[![Downloads](https://pepy.tech/badge/espnet_model_zoo)](https://pepy.tech/project/espnet_model_zoo)
[![GitHub license](https://img.shields.io/github/license/espnet/espnet_model_zoo.svg)](https://github.com/espnet/espnet_model_zoo)
![CI](https://github.com/espnet/espnet_model_zoo/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/espnet/espnet_model_zoo/branch/master/graph/badge.svg)](https://codecov.io/gh/espnet/espnet_model_zoo)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Utilities managing the pretrained models created by [ESPnet](https://github.com/espnet/espnet).

- Zenodo community: https://zenodo.org/communities/espnet/
- Registered models: [table.csv](espnet_model_zoo/table.csv)

## Install

```
pip install torch
pip install espnet_model_zoo
```

## Obtain a model

```python
from espnet_model_zoo.downloader import ModelDownloader
d = ModelDownloader("~/.cache/espnet")  # Specify cachedir
d = ModelDownloader()  # <module_dir> is used as cachedir by default
```

To obtain a model, you need to give a model in the form of `<user_name/model_name>`

```python
>>> d.download_and_unpack("kamo-naoyuki/mini_an4_asr_train_raw_bpe_valid.acc.best")
{"asr_train_config": <config path>, "asr_model_file": <model path>, ...}
```

Note that if the model already exists, you can skip downloading and unpacking.

You can also get a model with certain conditions.

```python
d.download_and_unpack(task="asr", corpus="wsj")
```

If multiple models are found with the condition, the last model is selected.
You can also specify the condition using "version" option.

```python
d.download_and_unpack(task="asr", corpus="wsj", version=-1)  # Get the last model
d.download_and_unpack(task="asr", corpus="wsj", version=-2)  # Get previous model
```

You can also obtain it from the URL directly.

```python
d.download_and_unpack("https://zenodo.org/record/...")
```

## Query model names

You can view the model names from our Zenodo community, https://zenodo.org/communities/espnet/, 
or using `query()`.  All information are written in [table.csv](espnet_model_zoo/table.csv). 

```python
d.query("name")
```

You can also show them with specifying certain conditions.

```python
d.query("name", task="asr")
```

## Command line tools

- `espnet_model_zoo_query`

    ```sh
    # Query model name
    espnet_model_zoo_query --condition task=asr --condition corpus=wsj 
    # Query the other key
    espnet_model_zoo_query --key url --condition task=asr --condition corpus=wsj 
    ```
- `espnet_model_zoo_download`

    ```sh
    espnet_model_zoo_download <model_name>  # Print the path of the downloaded file
    espnet_model_zoo_download --unpack true <model_name>   # Print the path of unpacked files
    ```
- `espnet_model_zoo_upload`

    ```sh
    export ACCESS_TOKEN=<access_token>
    espnet_zenodo_upload \
        --file <packed_model> \
        --title <title> \
        --description <description> \
        --creator_name <your-git-account>
    ```

## Use a pretrained model for inference

### ASR

```python
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
d = ModelDownloader()
speech2text = Speech2Text(**d.download_and_unpack("user_name/model_name"))

speech, rate = soundfile.read("speech.wav")
speech2text(speech)
[(text, token, token_int, hypothesis object), ...]
```

### TTS

```python
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech
d = ModelDownloader()
text2speech = Text2Speech(**d.download_and_unpack("user_name/model_name"))

retval = text2speech("foobar")
speech = retval[0]
soundfile.write("out.wav", speech.numpy(), text2speech.fs, "PCM_16")
```

## Register your model

1. Upload your model to Zenodo

    You need to [signup to Zenodo](https://zenodo.org/) and [create an access token](https://zenodo.org/account/settings/applications/tokens/new/) to upload models.
    You can upload your own model by using `espnet_model_zoo_upload` command freely, 
    but we normally upload a model using [recipes](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE).

1. Create a Pull Request to modify [table.csv](espnet_model_zoo/table.csv)

    You need to append your record at the last line.
1. (Administrator does) Increment the third version number of [setup.py](setup.py), e.g. 0.0.3 -> 0.0.4
1. (Administrator does) Release new version


## Update your model

If your model has some troubles, please modify the record at Zenodo directly or reupload a corrected file using `espnet_zenodo_upload` as another record.
