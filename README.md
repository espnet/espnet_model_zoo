# ESPnet Model Zoo

[![PyPI version](https://badge.fury.io/py/espnet-model-zoo.svg)](https://badge.fury.io/py/espnet-model-zoo)
[![Python Versions](https://img.shields.io/pypi/pyversions/espnet_model_zoo.svg)](https://pypi.org/project/espnet_model_zoo/)
[![Downloads](https://pepy.tech/badge/espnet_model_zoo)](https://pepy.tech/project/espnet_model_zoo)
[![GitHub license](https://img.shields.io/github/license/espnet/espnet_model_zoo.svg)](https://github.com/espnet/espnet_model_zoo)
[![Unitest](https://github.com/espnet/espnet_model_zoo/workflows/Unitest/badge.svg)](https://github.com/espnet/espnet_model_zoo/actions?query=workflow%3AUnitest)
[![Model test](https://github.com/espnet/espnet_model_zoo/workflows/Model%20test/badge.svg)](https://github.com/espnet/espnet_model_zoo/actions?query=workflow%3A%22Model+test%22)
[![codecov](https://codecov.io/gh/espnet/espnet_model_zoo/branch/master/graph/badge.svg)](https://codecov.io/gh/espnet/espnet_model_zoo)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Utilities managing the pretrained models created by [ESPnet](https://github.com/espnet/espnet). This function is inspired by the [Asteroid pretrained model function](https://github.com/mpariente/asteroid/blob/master/docs/source/readmes/pretrained_models.md).

- **From version 0.1.0, the huggingface models can be also used**: https://huggingface.co/models?filter=espnet
- Zenodo community: https://zenodo.org/communities/espnet/
- Registered models: [table.csv](espnet_model_zoo/table.csv)

## Install

```
pip install torch
pip install espnet_model_zoo
```

## Python API for inference
`model_name` in the following section should be `huggingface_id` or one of the tags in the [table.csv](espnet_model_zoo/table.csv).
Or you can directly provide zenodo URL (e.g., `https://zenodo.org/record/xxxxxxx/files/hogehoge.zip?download=1`).

### ASR

```python
import soundfile
from espnet2.bin.asr_inference import Speech2Text
speech2text = Speech2Text.from_pretrained(
    "model_name",
    # Decoding parameters are not included in the model file
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.3,
    lm_weight=0.5,
    penalty=0.0,
    nbest=1
)
# Confirm the sampling rate is equal to that of the training corpus.
# If not, you need to resample the audio data before inputting to speech2text
speech, rate = soundfile.read("speech.wav")
nbests = speech2text(speech)

text, *_ = nbests[0]
print(text)
```

### TTS

```python
import soundfile
from espnet2.bin.tts_inference import Text2Speech
text2speech = Text2Speech.from_pretrained("model_name")
speech = text2speech("foobar")["wav"]
soundfile.write("out.wav", speech.numpy(), text2speech.fs, "PCM_16")
```

### Speech separation

```python
import soundfile
from espnet2.bin.enh_inference import SeparateSpeech
separate_speech = SeparateSpeech.from_pretrained(
    "model_name",
    # for segment-wise process on long speech
    segment_size=2.4,
    hop_size=0.8,
    normalize_segment_scale=False,
    show_progressbar=True,
    ref_channel=None,
    normalize_output_wav=True,
)
# Confirm the sampling rate is equal to that of the training corpus.
# If not, you need to resample the audio data before inputting to speech2text
speech, rate = soundfile.read("long_speech.wav")
waves = separate_speech(speech[None, ...], fs=rate)
```

This API allows processing both short audio samples and long audio samples. For long audio samples, you can set the value of arguments segment_size, hop_size (optionally normalize_segment_scale and show_progressbar) to perform segment-wise speech enhancement/separation on the input speech. Note that the segment-wise processing is disabled by default.


<details><summary>For old ESPnet (<=10.1) </summary><div>

### ASR

```python
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
d = ModelDownloader()
speech2text = Speech2Text(
    **d.download_and_unpack("model_name"),
    # Decoding parameters are not included in the model file
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.3,
    lm_weight=0.5,
    penalty=0.0,
    nbest=1
)
```

### TTS

```python
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech
d = ModelDownloader()
text2speech = Text2Speech(**d.download_and_unpack("model_name"))
```

### Speech separation

```python
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.enh_inference import SeparateSpeech
d = ModelDownloader()
separate_speech = SeparateSpeech(
    **d.download_and_unpack("model_name"),
    # for segment-wise process on long speech
    segment_size=2.4,
    hop_size=0.8,
    normalize_segment_scale=False,
    show_progressbar=True,
    ref_channel=None,
    normalize_output_wav=True,
)
```
</div></details>


## Instruction for ModelDownloader

```python
from espnet_model_zoo.downloader import ModelDownloader
d = ModelDownloader("~/.cache/espnet")  # Specify cachedir
d = ModelDownloader()  # <module_dir> is used as cachedir by default
```

To obtain a model, you need to give a `huggingface_id`model` or a tag , which is listed in [table.csv](espnet_model_zoo/table.csv).

```python
>>> d.download_and_unpack("kamo-naoyuki/mini_an4_asr_train_raw_bpe_valid.acc.best")
{"asr_train_config": <config path>, "asr_model_file": <model path>, ...}
```

You can specify the revision if it's huggingface_id giving with `@`:

```python
>>> d.download_and_unpack("kamo-naoyuki/mini_an4_asr_train_raw_bpe_valid.acc.best@<revision>")
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

If you need to use a local model file using this API, you can also give it.

```python
d.download_and_unpack("./some/where/model.zip")
```

In this case, the contents are also expanded in the cache directory,
but the model is identified by the file path,
so if you move the model to somewhere and unpack again,
it's treated as another model,
thus the contents are expanded again at another place.

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
    espnet_model_zoo_query task=asr corpus=wsj
    # Show all model name
    espnet_model_zoo_query
    # Query the other key
    espnet_model_zoo_query --key url task=asr corpus=wsj
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

## Use pretrained model in ESPnet recipe

```sh
# e.g. ASR WSJ task
git clone https://github.com/espnet/espnet
pip install -e .
cd egs2/wsj/asr1
./run.sh --skip_data_prep false --skip_train true --download_model kamo-naoyuki/wsj
```

## Register your model

### Huggingface
1. Upload your model using huggingface API

    1. (if you do not have an HF hub account) Go to https://huggingface.co and create an HF account by clicking a `sign up` bottun below.
    ![image](https://user-images.githubusercontent.com/11741550/147585941-af1a7e88-934e-4e24-b30e-4b120dbc023a.png)
    2. From a `new model` link in the profile, create a new model repository. Please include a recipe name (e.g., aidatatang_200zh) and model info (e.g., conformer) in the repository name
    ![image](https://user-images.githubusercontent.com/11741550/147586093-51c98c53-6d23-45a0-b359-14a4489cc970.png)
    3. In the espnet recipe, execulte the following command:
    ```
    ./run.sh --stage 15 --skip_upload_hf false --hf_repo sw005320/aidatatang_200zh_conformer
    ```
    4. Please follow the instruction (e.g., type the HF Username/Password)
    5. If it works succesfully, you can get the following messages
    ![image](https://user-images.githubusercontent.com/11741550/147586699-a3bb5a49-8b59-417d-b376-4d1ec270fb71.png)

1. Create a Pull Request to modify [table.csv](espnet_model_zoo/table.csv)

    The model can be registered in [table.csv](https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv).
    Then, the model will be tested in the CI.
    Note that, unlike the zenodo case, you don't need to add the URL because huggingface_id itself can specify the model file, so please fill the value as `https://huggingface.co/`.

    e.g. `table.csv`

    ```
    ...
    aidatatang_200zh,asr,sw005320/aidatatang_200zh_conformer,https://huggingface.co/,16000,zh,,,,,true
    ```
1. (Administrator does) Increment the third version number of [setup.py](setup.py), e.g. 0.0.3 -> 0.0.4
1. (Administrator does) Release new version


### Zenodo (Obsolete)

1. Upload your model to Zenodo

    You need to [signup to Zenodo](https://zenodo.org/) and [create an access token](https://zenodo.org/account/settings/applications/tokens/new/) to upload models.
    You can upload your own model by using `espnet_model_zoo_upload` command freely,
    but we normally upload a model using [recipes](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE).

1. Create a Pull Request to modify [table.csv](espnet_model_zoo/table.csv)

    You need to append your record at the last line.
1. (Administrator does) Increment the third version number of [setup.py](setup.py), e.g. 0.0.3 -> 0.0.4
1. (Administrator does) Release new version
