"""Utilities for huggingface hub."""
from filelock import FileLock
import os
from typing import Dict
import yaml

from huggingface_hub import snapshot_download


META_YAML_FILENAME = "meta.yaml"


def setvalue(d: Dict, key: str, new_dir: str):
    if not isinstance(d[key], str):
        return
    v = d[key]
    if v is not None and any(v.startswith(prefix) for prefix in ["exp", "data"]):
        new_value = os.path.join(new_dir, v)
        d[key] = new_value


def hf_rewrite_yaml(yaml_file: str, cached_dir: str):
    """hf_rewrite_yaml."""
    touch_path = yaml_file + ".touch"
    lock_path = yaml_file + ".lock"

    with FileLock(lock_path):
        if not os.path.exists(touch_path):

            with open(yaml_file, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)

            for key in d:
                if isinstance(d[key], dict):
                    for skey in d[key]:
                        setvalue(d[key], skey, cached_dir)
                else:
                    setvalue(d, key, cached_dir)

            with open(yaml_file, "w", encoding="utf-8") as fw:
                yaml.safe_dump(d, fw)

            with open(touch_path, "a"):
                os.utime(touch_path, None)


def from_huggingface(huggingface_id: str):
    """Dowload a pretrained model stored at HuggingFace Hub.

    Args:
        huggingface_id (str): model id from the huggingface.co model hub
            e.g. ``"julien-c/mini_an4_asr_train_raw_bpe_valid"``
            and  ``julien-c/model@main`` supports specifying a commit/branch/tag.

    Returns:
        Dict with updated initialization files.

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
