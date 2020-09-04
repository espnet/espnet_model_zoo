import soundfile

import numpy as np

from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.tts_inference import Text2Speech
from espnet_model_zoo.downloader import ModelDownloader

speech = soundfile.read("watashiwakamodesu.wav")


def _asr(model_name):
    d = ModelDownloader()
    speech2text = Speech2Text(**d.download_and_unpack(model_name))
    nbests = speech2text(speech)
    text, *_ = nbests[0]
    assert isinstance(text, str)
    d.clean_cache(model_name)


def _tts(model_name):
    d = ModelDownloader()
    text2speech = Text2Speech(**d.download_and_unpack(model_name))
    speech, *_ = text2speech("foo")
    assert isinstance(speech, np.ndarray)
    d.clean_cache(model_name)


def test_model():
    d = ModelDownloader()
    tasks = ["asr", "tts"]

    for task in tasks:
        for model_name in d.query(task=task):
            if d.query("valid", name=model_name)[0] == "false":
                continue

            if task == "asr":
                _asr(model_name)
            elif task == "tts":
                _tts(model_name)
            else:
                raise NotImplementedError(f"task={task}")
