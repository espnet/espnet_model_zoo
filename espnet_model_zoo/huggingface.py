"""Utilities for huggingface hub."""
from filelock import FileLock
import functools
import os
from typing import Any
from typing import Dict
import yaml

from huggingface_hub import snapshot_download


META_YAML_FILENAME = "meta.yaml"


def nested_dict_get(dictionary: Dict, dotted_key: str):
    """nested_dict_get."""
    keys = dotted_key.split(".")
    return functools.reduce(lambda d, key: d.get(key) if d else None, keys, dictionary)


def nested_dict_set(dictionary: Dict, dotted_key: str, v: Any):
    """nested_dict_set."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, {})
    dictionary[keys[-1]] = v


def hf_rewrite_yaml(yaml_file: str, cached_dir: str):
    """hf_rewrite_yaml."""
    touch_path = yaml_file + ".touch"
    lock_path = yaml_file + ".lock"

    with FileLock(lock_path):
        if not os.path.exists(touch_path):

            with open(yaml_file, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)

            for key in d:
                print(d[key], type(d[key]))
                if not (isinstance(d[key], str) or isinstance(d[key], dict)):
                    continue
                if isinstance(d[key], dict):
                    
                v = nested_dict_get(d, key)
                if v is not None and any(
                    v.startswith(prefix) for prefix in ["exp", "data"]
                ):
                    new_value = os.path.join(cached_dir, v)
                    nested_dict_set(d, key, new_value)
                    print(new_value)
            with open(yaml_file, "w", encoding="utf-8") as fw:
                yaml.safe_dump(d, fw)

            with open(touch_path, "a"):
                os.utime(touch_path, None)


def from_huggingface(huggingface_id: str):
    """Instantiate a DiarizeSpeech model from a local packed archive or a model id

    Args:
        huggingface_id (str): model id from the huggingface.co model hub
            e.g. ``"julien-c/mini_an4_asr_train_raw_bpe_valid"``
            and  ``julien-c/model@main`` supports specifying a commit/branch/tag.

    Returns:
        instance of DiarizeSpeech

    """

    if "@" in huggingface_id:
        huggingface_id = huggingface_id.split("@")[0]
        revision = huggingface_id.split("@")[1]
    else:
        huggingface_id = huggingface_id
        revision = None
    cached_dir = snapshot_download(
        huggingface_id, revision=revision, library_name="espnet"
    )

    meta_yaml_path = os.path.join(cached_dir, META_YAML_FILENAME)
    with open(meta_yaml_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    assert isinstance(d, dict), type(d)

    yaml_files = d["yaml_files"]
    files = d["files"]
    assert isinstance(yaml_files, dict), type(yaml_files)
    assert isinstance(files, dict), type(files)
    inputs = {}
    for key, value in list(yaml_files.items()) + list(files.items()):
        inputs[key] = os.path.join(cached_dir, value)
        if key in yaml_files.keys():
            # Rewrite paths inside yaml
            hf_rewrite_yaml(inputs[key], cached_dir)
    return inputs
