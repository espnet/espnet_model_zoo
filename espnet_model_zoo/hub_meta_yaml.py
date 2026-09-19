#!/usr/bin/env python3
"""Give the espnet organisation's hand-uploaded models a meta.yaml.

    python -m espnet_model_zoo.hub_meta_yaml plan  [--out meta_plan.csv]
    python -m espnet_model_zoo.hub_meta_yaml apply --plan meta_plan.csv [--dry-run]

A model published with `espnet2.bin.pack` carries meta.yaml at its root: it
names the training config and the checkpoint under the key names the task's
inference class takes, and ModelDownloader hands those straight to the
constructor. A repository uploaded by hand holds the same files with nothing
saying which is which, so `download_and_unpack` refuses it and the user gets a
FileNotFoundError from inside the cache.

`plan` works out, per repository, which file is the config, which is the
checkpoint and what the task is, and writes one row per model with its
evidence and anything that blocks it. `apply` uploads meta.yaml for the rows
that are ready - one small text file per repository; no weight is re-uploaded,
which is why this is cheap enough to do for every model at once. `apply` needs
a token with write access to the organisation (`hf auth login`); `plan` needs
no token.

What meta.yaml cannot fix is named in the `blockers` column: a config that
refers to a bpemodel or a normalisation statistics file the repository does
not contain, or a checkpoint that is a dangling symlink. Those need the file,
not this script. Nor does meta.yaml make an old model build against current
espnet: a config written for a since-removed argument still fails, and the
plan says nothing about that - load the model to find out.
"""

import argparse
import csv
import posixpath
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import requests
import yaml

ORG = "espnet"
API = "https://huggingface.co/api"
TIMEOUT = 120

# From espnet2/bin/pack.py: the key names are the inference class's arguments,
# so the dict ModelDownloader returns can be splatted into the constructor.
TASK_KEYS = {
    "asr": ("asr_train_config", "asr_model_file"),
    "st": ("st_train_config", "st_model_file"),
    "s2t": ("s2t_train_config", "s2t_model_file"),
    "s2st": ("s2st_train_config", "s2st_model_file"),
    "enh_s2t": ("enh_s2t_train_config", "enh_s2t_model_file"),
    "ssl": ("ssl_train_config", "ssl_model_file"),
    "cls": ("classification_train_config", "classification_model_file"),
    "tts": ("train_config", "model_file"),
    "enh": ("train_config", "model_file"),
    "diar": ("train_config", "model_file"),
    "svs": ("train_config", "model_file"),
    "spk": ("train_config", "model_file"),
    "lid": ("train_config", "model_file"),
    "codec": ("train_config", "model_file"),
    # No PackedContents class of their own; the generic pair is what their
    # Task class takes.
    "vad": ("train_config", "model_file"),
    "speechlm": ("train_config", "model_file"),
}

# Config values that inference reads from disk. Everything else a training
# config mentions - wav.scp, shape files, the recipe's own conf/*.yaml - is
# not needed to load the model.
INFERENCE_INPUTS: Sequence[Tuple[str, ...]] = (
    ("token_list",),
    ("bpemodel",),
    ("non_linguistic_symbols",),
    ("normalize_conf", "stats_file"),
    ("pitch_normalize_conf", "stats_file"),
    ("energy_normalize_conf", "stats_file"),
    ("feats_extract_conf", "stats_file"),
    ("subword_model",),
    ("tokenizer_conf", "model"),
)

# "HuggingFaceTB/SmolLM-1.7B" is a Hub repository the tokenizer pulls at load
# time, and its tail reads like an extension, so a value counts as a file only
# when its suffix is one a repository actually holds.
FILE_SUFFIXES = frozenset(
    {".model", ".npz", ".txt", ".yaml", ".yml", ".json", ".mdl", ".vocab", ".scp"}
)

NOT_MODEL_FILES = frozenset({".gitattributes", "README.md"})

# A checkpoint file small enough to be a symlink stored as its target's name.
SYMLINK_BYTES = 4096


class FetchError(RuntimeError):
    """The Hub could not be read. Never confused with "nothing found"."""


def _get(url: str):
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise FetchError(f"{url}: {e}") from e


def _get_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        raise FetchError(f"{url}: {e}") from e


def task_of(config: dict, output_dir: str) -> str:
    """Prefer what the config is made of; fall back to the experiment name."""
    if "tts" in config:
        return "tts"
    if "svs" in config:
        return "svs"
    if "separator" in config or "diffusion_model" in config:
        return "enh"
    if "attractor" in config:
        return "diar"
    if "corelm" in config:
        return "speechlm"
    if "codec" in config and "codec_conf" in config:
        return "codec"
    if "pooling" in config and "projector" in config:
        return "spk"
    if "pred_masked_weight" in config:
        return "ssl"
    name = posixpath.basename(output_dir or "")
    for prefix in (
        "s2t",
        "asr",
        "st",
        "diar",
        "vad",
        "svs",
        "spk",
        "tts",
        "enh",
        "lid",
        "cls",
        "mt",
        "uasr",
        "ssl",
        "codec",
    ):
        if name.startswith(prefix + "_"):
            return prefix
    if "ctc_conf" in config and "decoder" in config:
        return "asr"
    return ""


def _config_rank(path: str) -> tuple:
    """A training config, not a statistics or decoding one."""
    low = path.lower()
    return (
        "logdir" in low or "stats" in low,
        "decode" in low or "inference" in low,
        posixpath.basename(low) != "config.yaml",
        low.count("/"),
        path,
    )


def checkpoint_rank(path: str) -> tuple:
    """Order checkpoints by how much they look like the published model.

    `checkpoint.pth` is the trainer's resume state - optimizer included - and
    is not what an inference class loads, so it sorts last.
    """
    name = posixpath.basename(path)
    epoch = re.match(r"(\d+)epoch\.pth$", name)
    return (
        name == "checkpoint.pth",
        not name.startswith(("valid.", "train.")),
        ".ave" not in name,
        -int(epoch.group(1)) if epoch else 0,
        name,
    )


def symlink_target(path: str, text: str) -> Optional[str]:
    """The checkpoint a symlink-as-text points at, or None if it is not one.

    espnet writes `valid.acc.ave.pth` as a symlink to the averaged file;
    uploaded by hand, the Hub stores the 24-byte target name as the content,
    and torch.load on it fails on a text file.
    """
    text = text.strip()
    if "\n" in text or not text.endswith((".pth", ".pt")):
        return None
    return posixpath.normpath(posixpath.join(posixpath.dirname(path), text))


def _value_at(config: dict, keys: Sequence[str]):
    node = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def missing_inputs(config: dict, files: Sequence[str]) -> List[str]:
    """Files the config needs at load time that the repository does not have."""
    present = set(files)
    missing = []
    for keys in INFERENCE_INPUTS:
        value = _value_at(config, keys)
        # token_list is a path only when it is a path: written out it is the
        # vocabulary itself, and "<sos/eos>" is a token, not a file.
        if not isinstance(value, str) or "/" not in value:
            continue
        if posixpath.splitext(value)[1] not in FILE_SUFFIXES:
            continue
        if value not in present and value.lstrip("./") not in present:
            missing.append(f"{'.'.join(keys)}={value}")
    return missing


def choose_pair(files: Sequence[str]) -> Tuple[str, str, bool]:
    """The config and checkpoint a meta.yaml should name.

    Returns (config, checkpoint, split), where split says the two came from
    different directories - some repositories keep an averaged checkpoint in
    the experiment directory next door, and which one belongs to which config
    is not this script's call to make silently.
    """
    configs = [
        f for f in files if posixpath.basename(f) in ("config.yaml", "config.yml")
    ]
    checkpoints = [f for f in files if f.endswith((".pth", ".pt"))]
    if not configs or not checkpoints:
        return "", "", False
    by_directory: Dict[str, Tuple[List[str], List[str]]] = {}
    for f in configs:
        by_directory.setdefault(posixpath.dirname(f), ([], []))[0].append(f)
    for f in checkpoints:
        by_directory.setdefault(posixpath.dirname(f), ([], []))[1].append(f)
    complete = {d: v for d, v in by_directory.items() if v[0] and v[1]}
    if not complete:
        return (
            sorted(configs, key=_config_rank)[0],
            sorted(checkpoints, key=checkpoint_rank)[0],
            True,
        )
    directory = sorted(complete, key=lambda d: _config_rank(complete[d][0][0]))[0]
    return (
        sorted(complete[directory][0], key=_config_rank)[0],
        sorted(complete[directory][1], key=checkpoint_rank)[0],
        False,
    )


def meta_yaml(task: str, config: str, checkpoint: str) -> str:
    """The file itself: the two keys ModelDownloader reads."""
    config_key, file_key = TASK_KEYS[task]
    return yaml.safe_dump(
        {"files": {file_key: checkpoint}, "yaml_files": {config_key: config}},
        default_flow_style=False,
        sort_keys=True,
    )


def plan_one(model_id: str) -> Dict[str, str]:
    """Everything a row needs: what to write, and what stops it being written."""
    row = {
        "model": model_id,
        "task": "",
        "train_config": "",
        "model_file": "",
        "evidence": "",
        "blockers": "",
    }
    info = _get(f"{API}/models/{model_id}?blobs=true")
    siblings = info.get("siblings", [])
    files = [s["rfilename"] for s in siblings if not s["rfilename"].startswith(".")]
    sizes = {s["rfilename"]: s.get("size") for s in siblings}
    if not [f for f in files if f not in NOT_MODEL_FILES]:
        row["blockers"] = "empty repository"
        return row

    config, checkpoint, split = choose_pair(files)
    if not config or not checkpoint:
        row["blockers"] = "no config.yaml" if not config else "no checkpoint"
        return row

    size = sizes.get(checkpoint)
    if size is not None and size <= SYMLINK_BYTES:
        target = symlink_target(
            checkpoint,
            _get_text(f"https://huggingface.co/{model_id}/resolve/main/{checkpoint}"),
        )
        if target is None:
            row["blockers"] = f"{checkpoint} is {size} bytes and is not a checkpoint"
            return row
        if target not in files:
            row["blockers"] = f"{checkpoint} points at {target}, which is not here"
            return row
        checkpoint = target

    config_text = _get_text(f"https://huggingface.co/{model_id}/resolve/main/{config}")
    try:
        parsed = yaml.safe_load(config_text) or {}
    except yaml.YAMLError as e:
        row["blockers"] = f"{config} does not parse: {e}"
        return row
    if not isinstance(parsed, dict):
        row["blockers"] = f"{config} is not a mapping"
        return row

    task = task_of(parsed, str(parsed.get("output_dir", "")))
    row.update(task=task, train_config=config, model_file=checkpoint)
    row["evidence"] = f"output_dir={parsed.get('output_dir', '')}"
    blockers = []
    if not task:
        blockers.append("task undecided: fill in the task column")
    elif task not in TASK_KEYS:
        blockers.append(f"no packed-contents key names for task {task}")
    if split:
        blockers.append("config and checkpoint come from different directories")
    blockers += [f"missing {m}" for m in missing_inputs(parsed, files)]
    row["blockers"] = "; ".join(blockers)
    return row


def plan(out_path: str) -> None:
    models = _get(f"{API}/models?author={ORG}&full=true&limit=1000")
    unpacked = [
        m
        for m in models
        if not any(s["rfilename"] == "meta.yaml" for s in m.get("siblings", []))
    ]
    print(f"{len(models)} models, {len(unpacked)} without meta.yaml", file=sys.stderr)
    rows = []
    for i, m in enumerate(unpacked, 1):
        model_id = m["modelId"]
        try:
            rows.append(plan_one(model_id))
        except FetchError as e:
            rows.append(
                {
                    "model": model_id,
                    "task": "",
                    "train_config": "",
                    "model_file": "",
                    "evidence": "",
                    "blockers": f"fetch error, not decided: {e}",
                }
            )
        if i % 25 == 0:
            print(f"  {i}/{len(unpacked)}", file=sys.stderr)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "task",
                "train_config",
                "model_file",
                "evidence",
                "blockers",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    ready = sum(1 for r in rows if r["model_file"] and not r["blockers"])
    print(
        f"wrote {out_path}: {ready} ready, {len(rows) - ready} need a look",
        file=sys.stderr,
    )


def apply(plan_path: str, dry_run: bool) -> int:
    from huggingface_hub import HfApi

    with open(plan_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ready = [
        r
        for r in rows
        if r["train_config"] and r["model_file"] and r["task"] and not r["blockers"]
    ]
    print(f"{len(ready)} of {len(rows)} rows are ready", file=sys.stderr)
    api = HfApi()
    failed = []
    for row in ready:
        if row["task"] not in TASK_KEYS:
            failed.append((row["model"], f"unknown task {row['task']}"))
            continue
        content = meta_yaml(row["task"], row["train_config"], row["model_file"])
        if dry_run:
            print(f"--- {row['model']}\n{content}")
            continue
        try:
            api.upload_file(
                path_or_fileobj=content.encode(),
                path_in_repo="meta.yaml",
                repo_id=row["model"],
                repo_type="model",
                commit_message="Add meta.yaml so espnet_model_zoo can load this model",
            )
            print(f"uploaded {row['model']}")
        except Exception as e:  # the Hub's errors are many; each is one row
            failed.append((row["model"], str(e)))
    for model_id, why in failed:
        print(f"FAILED {model_id}: {why}", file=sys.stderr)
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="write the plan as CSV")
    p.add_argument("--out", default="meta_plan.csv")
    a = sub.add_parser("apply", help="upload meta.yaml for the ready rows")
    a.add_argument("--plan", default="meta_plan.csv")
    a.add_argument("--dry-run", action="store_true", help="print, upload nothing")
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan(args.out)
        return 0
    return apply(args.plan, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
