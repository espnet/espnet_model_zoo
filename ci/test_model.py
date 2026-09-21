"""Load every published model, and report every one that fails.

The whole table used to be one test, so the first failure hid the rest: an
`init: chainer` model stopped loading when espnet removed that choice in June,
the weekly run reported it in September, and nothing in the report said
whether it was alone or one of forty. Each model is a test of its own now, so
a run names every model that broke and the ones that still work are still
green.
"""

import os
import shutil

import numpy as np
import pytest
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
from espnet2.bin.tts_inference import Text2Speech

from espnet_model_zoo.downloader import ModelDownloader

CACHE = "downloads"


def _asr(model_name):
    d = ModelDownloader(CACHE)
    speech2text = Speech2Text(**d.download_and_unpack(model_name, quiet=True))
    speech = np.zeros((10000,), dtype=np.float32)
    nbests = speech2text(speech)
    text, *_ = nbests[0]
    assert isinstance(text, str)


def _asr_streaming(model_name):
    d = ModelDownloader(CACHE)
    speech2text = Speech2TextStreaming(**d.download_and_unpack(model_name, quiet=True))
    speech = np.zeros((10000,), dtype=np.float32)
    nbests = speech2text(speech)
    text, *_ = nbests[0]
    assert isinstance(text, str)


def _tts(model_name):
    d = ModelDownloader(CACHE)
    text2speech = Text2Speech(**d.download_and_unpack(model_name, quiet=True))
    inputs = {"text": "foo"}
    if text2speech.use_speech:
        inputs["speech"] = np.zeros((10000,), dtype=np.float32)
    if text2speech.use_spembs:
        inputs["spembs"] = np.zeros((text2speech.tts.spk_embed_dim,), dtype=np.float32)
    if text2speech.use_sids:
        inputs["sids"] = np.ones((1,), dtype=np.int64)
    if text2speech.use_lids:
        inputs["lids"] = np.ones((1,), dtype=np.int64)
    text2speech(**inputs)


LOAD = {"asr": _asr, "asr_stream": _asr_streaming, "tts": _tts}


def _published():
    """Every model the table calls valid, as a test case each.

    Read when the tests are collected, so a model added to table.csv is
    covered without touching this file.
    """
    d = ModelDownloader()
    for task in LOAD:
        for corpus in sorted(set(d.query("corpus", task=task))):
            for name in d.query(task=task, corpus=corpus):
                if d.query("valid", name=name)[0] == "false":
                    continue
                yield pytest.param(task, name, id=f"{task}-{name}")


@pytest.fixture()
def cache():
    """Download into a directory that goes away again.

    The models are tens of gigabytes together and the runner has less than
    that, so each is removed once it has been loaded. It was per corpus
    before, which is the same idea with a coarser broom.
    """
    yield CACHE
    shutil.rmtree(CACHE, ignore_errors=True)
    os.makedirs(CACHE, exist_ok=True)


@pytest.mark.parametrize("task, model_name", list(_published()))
def test_model(task, model_name, cache):
    LOAD[task](model_name)
