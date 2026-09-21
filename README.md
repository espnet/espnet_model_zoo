<div align="center">

# ESPnet Model Zoo

### Pretrained [ESPnet](https://github.com/espnet/espnet) models, and the tools that keep them loadable

[![PyPI](https://img.shields.io/pypi/v/espnet_model_zoo?color=%233775A9&logo=pypi&logoColor=white)](https://pypi.org/project/espnet_model_zoo/)
[![Python](https://img.shields.io/pypi/pyversions/espnet_model_zoo.svg)](https://pypi.org/project/espnet_model_zoo/)
[![Downloads](https://static.pepy.tech/badge/espnet_model_zoo/month)](https://pepy.tech/project/espnet_model_zoo)
[![License](https://img.shields.io/github/license/espnet/espnet_model_zoo.svg?color=blue)](./LICENSE)
[![Unitest](https://github.com/espnet/espnet_model_zoo/workflows/Unitest/badge.svg)](https://github.com/espnet/espnet_model_zoo/actions?query=workflow%3AUnitest)
[![Model test](https://github.com/espnet/espnet_model_zoo/workflows/Model%20test/badge.svg)](https://github.com/espnet/espnet_model_zoo/actions?query=workflow%3A%22Model+test%22)
[![codecov](https://codecov.io/gh/espnet/espnet_model_zoo/branch/master/graph/badge.svg)](https://codecov.io/gh/espnet/espnet_model_zoo)

**[Models on Hugging Face](https://huggingface.co/espnet)** ·
**[Registered models](espnet_model_zoo/table.csv)** ·
**[ESPnet](https://github.com/espnet/espnet)** ·
**[ESPnet docs](https://espnet.github.io/espnet/)**

</div>

______________________________________________________________________

`espnet_model_zoo` downloads a pretrained ESPnet model, unpacks it, and hands the
resulting paths to the matching `espnet2` inference class — so loading a model is one
`from_pretrained` call rather than a checkpoint, a config and a token list to wire up
yourself. The models live in the [espnet organization on Hugging
Face](https://huggingface.co/espnet): 671 of them as of 2026-09-21, 624 carrying a
`pipeline_tag` you can filter by. The design follows [Asteroid's pretrained model
function](https://github.com/mpariente/asteroid/blob/master/docs/source/readmes/pretrained_models.md).

> [!IMPORTANT]
> Upgrade to **0.1.11** or later. Releases before it resolved paths through every string
> in a packed config, including the vocabulary, so a token that happened to name
> something inside the cache — OWSM's `.` and `exp`, for instance — was rewritten into an
> absolute path and came back in the transcript. 0.1.11 stops doing it and repairs a
> cache an older version already damaged, on load, with no re-download.

## Install

```sh
pip install torch                # first, per https://pytorch.org/get-started/locally/
pip install espnet_model_zoo     # brings espnet with it
```

## Quick start

A model name is a Hugging Face id (`espnet/owsm_ctc_v4_1B`), a tag from
[table.csv](espnet_model_zoo/table.csv), a local `.zip`, or a Zenodo URL. Every task
follows the same `from_pretrained` shape:

```python
import soundfile
from espnet2.bin.asr_inference import Speech2Text

speech2text = Speech2Text.from_pretrained("model_name")
speech, rate = soundfile.read("speech.wav")   # at the model's training sample rate
text, *_ = speech2text(speech)[0]
print(text)
```

```python
import soundfile
from espnet2.bin.tts_inference import Text2Speech

text2speech = Text2Speech.from_pretrained("model_name")
speech = text2speech("foobar")["wav"]
soundfile.write("out.wav", speech.numpy(), text2speech.fs, "PCM_16")
```

```python
import soundfile
from espnet2.bin.enh_inference import SeparateSpeech

separate_speech = SeparateSpeech.from_pretrained("model_name")
speech, rate = soundfile.read("long_speech.wav")
waves = separate_speech(speech[None, ...], fs=rate)
```

Resample your audio to the rate the model was trained at; nothing does it for you.

<details>
<summary>Decoding and segmentation parameters</summary>

Decoding parameters are not stored in the model file, so pass them to
`from_pretrained`:

```python
speech2text = Speech2Text.from_pretrained(
    "model_name",
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.3,
    lm_weight=0.5,
    penalty=0.0,
    nbest=1,
)
```

`SeparateSpeech` handles both short and long audio. Segment-wise processing is off by
default; `segment_size` and `hop_size` turn it on, and `normalize_segment_scale` and
`show_progressbar` tune it:

```python
separate_speech = SeparateSpeech.from_pretrained(
    "model_name",
    segment_size=2.4,
    hop_size=0.8,
    normalize_segment_scale=False,
    show_progressbar=True,
    ref_channel=None,
    normalize_output_wav=True,
)
```

</details>

<details>
<summary>The API before ESPnet 0.10.1</summary>

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

</details>

## Find a model

Filter the [Hugging Face organization](https://huggingface.co/espnet) by task, or query
[table.csv](espnet_model_zoo/table.csv) locally:

```python
from espnet_model_zoo.downloader import ModelDownloader

d = ModelDownloader()
d.query("name")                    # every registered name
d.query("name", task="asr")        # narrowed by any column of table.csv
```

```sh
espnet_model_zoo_query                              # all names
espnet_model_zoo_query task=asr corpus=wsj          # narrowed
espnet_model_zoo_query --key url task=asr corpus=wsj
```

## Download and cache

```python
from espnet_model_zoo.downloader import ModelDownloader

d = ModelDownloader()                    # ~/.cache/espnet_model_zoo; Hugging Face
                                         # models go to the huggingface_hub cache
d = ModelDownloader("~/.cache/espnet")   # or choose the directory
```

`download_and_unpack` returns the paths an inference class needs, and skips the work if
the model is already there:

```python
>>> d.download_and_unpack("kamo-naoyuki/mini_an4_asr_train_raw_bpe_valid.acc.best")
{"asr_train_config": <config path>, "asr_model_file": <model path>, ...}
```

It takes the same four kinds of name as `from_pretrained`, plus a query:

```python
d.download_and_unpack("kamo-naoyuki/mini_an4_...@<revision>")  # a Hub revision
d.download_and_unpack("https://zenodo.org/record/...")         # a URL
d.download_and_unpack("./some/where/model.zip")                # a local file
d.download_and_unpack(task="asr", corpus="wsj")                # a query: last match
d.download_and_unpack(task="asr", corpus="wsj", version=-2)    # the one before it
```

A local file is unpacked into the cache too, and is identified by its path — move it and
unpack again and it is treated as a different model, expanded a second time.

If a model was uploaded to the Hub by hand rather than by a recipe, it has no `meta.yaml`
saying which file is the config and which is the checkpoint. `download_and_unpack` then
fails with a `RuntimeError` listing the repository's files, and you pass `train_config`
and `model_file` yourself. Tell us which model it was — repairing those in place is a
maintainer job, described in [MAINTAINING.md](MAINTAINING.md).

```sh
espnet_model_zoo_download <model_name>                # prints the downloaded file
espnet_model_zoo_download --unpack true <model_name>  # prints the unpacked files
```

## Use a model in an ESPnet recipe

```sh
# e.g. ASR WSJ task
git clone https://github.com/espnet/espnet
pip install -e .
cd egs2/wsj/asr1
./run.sh --skip_data_prep false --skip_train true --download_model kamo-naoyuki/wsj
```

## Publish your model

Upload from the recipe that trained it, then register it here.

1. Create a [Hugging Face account](https://huggingface.co) and a
   [new model repository](https://huggingface.co/new). Name it after the recipe and the
   model, e.g. `aidatatang_200zh_conformer`.
2. From the recipe, push the trained model:

   ```sh
   ./run.sh --stage 15 --skip_upload_hf false --hf_repo <user>/aidatatang_200zh_conformer
   ```

   The stage number is the upload stage of that task's pipeline — 15 for `asr1`, other
   tasks differ, so check `./run.sh --help`.
3. Open a pull request adding a row to
   [table.csv](https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv),
   so the model is covered by CI. A Hugging Face id identifies the model by itself, so
   the `url` column is just `https://huggingface.co/`:

   ```
   aidatatang_200zh,asr,sw005320/aidatatang_200zh_conformer,https://huggingface.co/,16000,zh,,,,,true
   ```
4. An administrator increments the third version number in [setup.py](setup.py) and
   releases.

<details>
<summary>Screenshots of the Hub steps</summary>

Creating an account:

![sign up](https://user-images.githubusercontent.com/11741550/147585941-af1a7e88-934e-4e24-b30e-4b120dbc023a.png)

Creating the model repository:

![new model](https://user-images.githubusercontent.com/11741550/147586093-51c98c53-6d23-45a0-b359-14a4489cc970.png)

A successful upload:

![success](https://user-images.githubusercontent.com/11741550/147586699-a3bb5a49-8b59-417d-b376-4d1ec270fb71.png)

</details>

<details>
<summary>Zenodo (obsolete)</summary>

1. Upload your model to Zenodo

    You need to [signup to Zenodo](https://zenodo.org/) and [create an access token](https://zenodo.org/account/settings/applications/tokens/new/) to upload models.
    You can upload your own model by using `espnet_model_zoo_upload` command freely,
    but we normally upload a model using [recipes](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE).

1. Create a Pull Request to modify [table.csv](espnet_model_zoo/table.csv)

    You need to append your record at the last line.
1. (Administrator does) Increment the third version number of [setup.py](setup.py), e.g. 0.0.3 -> 0.0.4
1. (Administrator does) Release new version

```sh
export ACCESS_TOKEN=<access_token>
espnet_model_zoo_upload \
    --file <packed_model> \
    --title <title> \
    --description <description> \
    --creator_name <your-git-account>
```

</details>

______________________________________________________________________

<div align="center">
Maintaining the Hugging Face organization: <a href="MAINTAINING.md">MAINTAINING.md</a> ·
Released under the <a href="./LICENSE">Apache 2.0 License</a>.
</div>
