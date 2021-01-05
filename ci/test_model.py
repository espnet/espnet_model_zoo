import os
import shutil

import numpy as np

from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.tts_inference import Text2Speech
from espnet_model_zoo.downloader import ModelDownloader


def _asr(model_name):
    d = ModelDownloader()
    speech2text = Speech2Text(**d.download_and_unpack(model_name))
    speech = np.zeros((10000,), dtype=np.float32)
    nbests = speech2text(speech)
    text, *_ = nbests[0]
    assert isinstance(text, str)


def _tts(model_name):
    d = ModelDownloader()
    text2speech = Text2Speech(**d.download_and_unpack(model_name))
    inputs = {"text": "foo"}
    if text2speech.use_speech:
        inputs["speech"] = np.zeros((10000,), dtype=np.float32)
    if text2speech.tts.spk_embed_dim is not None:
        inputs["spembs"] = np.zeros((text2speech.tts.spk_embed_dim,), dtype=np.float32)
    text2speech(**inputs)


def test_model():
    d = ModelDownloader("downloads")
    tasks = ["asr", "tts"]

    for task in tasks:
        for corpus in list(set(d.query("corpus", task=task))):
            for model_name in d.query(task=task, corpus=corpus):
                if d.query("valid", name=model_name)[0] == "false":
                    continue
                print(f"#### Test {model_name} ####")

                if task == "asr":
                    _asr(model_name)
                elif task == "tts":
                    _tts(model_name)
                else:
                    raise NotImplementedError(f"task={task}")

            # NOTE(kan-bayashi): remove and recreate cache dir to reduce the disk usage.
            shutil.rmtree("downloads")
            os.makedirs("downloads")
