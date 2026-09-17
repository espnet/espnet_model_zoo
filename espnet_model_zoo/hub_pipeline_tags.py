#!/usr/bin/env python3
"""Give the espnet organisation's Hub models a pipeline_tag.

    python -m espnet_model_zoo.hub_pipeline_tags plan  [--out plan.csv]
    python -m espnet_model_zoo.hub_pipeline_tags apply --plan plan.csv [--dry-run]

`plan` reads every model under the organisation that has no pipeline_tag and
infers one from, in order of trust, the constructor keys in its meta.yaml
(asr_train_config -> ASR), the task prefix of its exp/ directory, and words in
its name and tags. It writes one row per model with the evidence and leaves
the tag empty when nothing decides it, so the CSV can be edited before
`apply` pushes the tag into each model card's front matter with
huggingface_hub.metadata_update. `apply` needs a token with write access to
the organisation (`hf auth login`); `plan` needs no token.

A model without a pipeline_tag does not appear when the Hub is filtered by
task, and gets no task widget: 268 of the organisation's 667 models were in
that state on 2026-09-17.
"""

import argparse
import csv
import re
import sys
from typing import Dict, List, Optional, Tuple

import requests

ORG = "espnet"
API = "https://huggingface.co/api"

# espnet task name -> Hub pipeline_tag
TASK_TAG = {
    "asr": "automatic-speech-recognition",
    "asr_transducer": "automatic-speech-recognition",
    "s2t": "automatic-speech-recognition",
    "uasr": "automatic-speech-recognition",
    "enh_asr": "automatic-speech-recognition",
    "enh_s2t": "automatic-speech-recognition",
    "tts": "text-to-speech",
    "gan_tts": "text-to-speech",
    "svs": "text-to-speech",
    "gan_svs": "text-to-speech",
    "speechlm": "text-to-speech",
    "enh": "audio-to-audio",
    "enh_tse": "audio-to-audio",
    "codec": "audio-to-audio",
    "gan_codec": "audio-to-audio",
    "s2st": "audio-to-audio",
    "spk": "audio-classification",
    "lid": "audio-classification",
    "slu": "audio-classification",
    "cls": "audio-classification",
    "diar": "voice-activity-detection",
    "asvspoof": "audio-classification",
    "st": "translation",
    "mt": "translation",
    "lm": "text-generation",
}

# Words in a model name or its tags that decide the task when the files do not.
# Order matters: the first pattern that matches wins. Task words only - a
# corpus (librispeech) or an SSL backbone (hubert, wavlm) in the name says
# nothing about what the model does, and a first version of this list tagged
# a vocoder, a TTS model and a speech-quality predictor as ASR that way.
NAME_PATTERNS: List[Tuple[str, str]] = [
    # not a task model at all: leave blank for a person to decide
    (r"universa|intermediates|opuslm|speechlm_unified", ""),
    (r"vocoder|hifigan|wavegan|melgan", "audio-to-audio"),
    # joint enhancement + recognition is a recogniser; must precede the enh rule
    (r"(^|[\s_/-])enh_(asr|s2t|st)([\s_/-]|$)", "automatic-speech-recognition"),
    (r"(^|[\s_/-])(vad|diar)([\s_/-]|$)|diariz", "voice-activity-detection"),
    (
        r"(^|[\s_/-])(tts|svs|vits|jets|fastspeech|tacotron)([\s_/-]|$)|singing|opencpop|kising|utagoe",  # noqa: E501
        "text-to-speech",
    ),  # noqa: E501
    (
        r"(^|[\s_/-])(enh|tse)([\s_/-]|$)|codec|soundstream|encodec|(^|[\s_/-])dac([\s_/-]|$)|tfgridnet|snsd|wsj0_2mix|librimix|whamr?([\s_/-]|$)|separation|(^|[\s_/-])dns([\s_/-]|$)",  # noqa: E501
        "audio-to-audio",
    ),  # noqa: E501
    (
        r"(^|[\s_/-])(st|mt|s2st)([\s_/-]|$)|must-?c|iwslt|covost|translation",
        "translation",
    ),  # noqa: E501
    (
        r"(^|[\s_/-])(asr|s2t|owsm|owls|whisper|transducer)",
        "automatic-speech-recognition",
    ),  # noqa: E501
    (
        r"(^|[\s_/-])(spk|lid|slu|cls\d?|asvspoof)([\s_/-]|$)|geolid|voxceleb|cnceleb|rawnet|ecapa|voxlingua|slurp|(^|[\s_/-])fsc([\s_/-]|$)|snips|intent|emotion|iemocap|dcase|esc-?50|audioset|beats",  # noqa: E501
        "audio-classification",
    ),  # noqa: E501
    (r"(^|[\s_/-])lm([\s_/-]|$)|language_model", "text-generation"),
    # a bare SSL encoder - the Hub's convention for representation models - but
    # only when the name says it is one; "x_hubert" may be an ASR fine-tune
    (
        r"(hubert|wavlm|wav2vec|xls-?r|data2vec).*(ssl|pretrain|iter\d|ckpt|dummy)|(ssl|pretrain).*(hubert|wavlm|wav2vec)|xeus|wavlablm",  # noqa: E501
        "feature-extraction",
    ),  # noqa: E501
    (r"hubert|wavlm|wav2vec|xls-?r|data2vec", ""),
]


def _get(url: str, timeout: float = 60) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def _get_text(url: str, timeout: float = 60) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout)
        return r.text if r.ok else None
    except requests.RequestException:
        return None


def infer_from_meta(meta_yaml: Optional[str]) -> Optional[Tuple[str, str]]:
    """The keys pack.py writes name the task: asr_train_config, tts_model_file..."""
    if not meta_yaml:
        return None
    for key in re.findall(
        r"^\s+([a-z_0-9]+?)_(?:train_config|model_file):", meta_yaml, re.M
    ):
        task = key.replace("_train", "")
        if key in ("train", "model"):
            continue  # tts/enh/spk pack as train_config/model_file: no task in the key
        if task in TASK_TAG:
            return TASK_TAG[task], f"meta.yaml key {key}_*"
    return None


def infer_from_files(files: List[str]) -> Optional[Tuple[str, str]]:
    """exp/<task>_train_.../ and exp/<task>_stats_... carry the task prefix."""
    for f in files:
        m = re.match(r"exp/([a-z_0-9]+?)_(?:train|stats)", f)
        if m and m.group(1) in TASK_TAG:
            return TASK_TAG[m.group(1)], f"file {f}"
    return None


def infer_from_name(model_id: str, tags: List[str]) -> Optional[Tuple[str, str]]:
    text = (model_id.split("/", 1)[-1] + " " + " ".join(tags)).lower()
    for pattern, tag in NAME_PATTERNS:
        m = re.search(pattern, text)
        if m:
            if not tag:
                return "", f"left for review: name matches {m.group(0)!r}"
            return tag, f"name/tags match {m.group(0)!r}"
    return None


def infer(model_id: str, tags: List[str], files: List[str], meta_yaml: Optional[str]):
    for source in (
        lambda: infer_from_meta(meta_yaml),
        lambda: infer_from_files(files),
        lambda: infer_from_name(model_id, tags),
    ):
        hit = source()
        if hit:
            return hit
    return "", "no evidence"


def plan(out_path: str) -> None:
    models = _get(f"{API}/models?author={ORG}&limit=1000") or []
    untagged = [m for m in models if not m.get("pipeline_tag")]
    print(
        f"{len(models)} models, {len(untagged)} without pipeline_tag", file=sys.stderr
    )
    rows: List[Dict[str, str]] = []
    for i, m in enumerate(untagged, 1):
        mid = m["modelId"]
        info = _get(f"{API}/models/{mid}") or {}
        files = [s["rfilename"] for s in info.get("siblings", [])]
        meta = (
            _get_text(f"https://huggingface.co/{mid}/raw/main/meta.yaml")
            if "meta.yaml" in files
            else None
        )
        tag, evidence = infer(mid, m.get("tags", []), files, meta)
        rows.append(
            {
                "model": mid,
                "pipeline_tag": tag,
                "evidence": evidence,
                "downloads": str(m.get("downloads", "")),
            }
        )
        if i % 25 == 0:
            print(f"  {i}/{len(untagged)}", file=sys.stderr)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["model", "pipeline_tag", "evidence", "downloads"]
        )
        w.writeheader()
        w.writerows(rows)
    decided = sum(1 for r in rows if r["pipeline_tag"])
    print(
        f"wrote {out_path}: {decided} decided, {len(rows) - decided} left blank for review",  # noqa: E501
        file=sys.stderr,
    )


def apply(plan_path: str, dry_run: bool) -> None:
    from huggingface_hub import metadata_update

    with open(plan_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["pipeline_tag"]]
    print(
        f"{len(rows)} models to tag" + (" (dry run)" if dry_run else ""),
        file=sys.stderr,
    )
    for r in rows:
        print(f"{r['model']}: {r['pipeline_tag']}   [{r['evidence']}]")
        if not dry_run:
            # overwrite=False: a tag someone set by hand in the meantime wins.
            metadata_update(
                r["model"],
                {"pipeline_tag": r["pipeline_tag"]},
                repo_type="model",
                overwrite=False,
                commit_message=f"Set pipeline_tag: {r['pipeline_tag']}",
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser(
        "plan", help="infer a tag for every untagged model; write a CSV"
    )
    sp.add_argument("--out", default="pipeline_tags.csv")
    sa = sub.add_parser("apply", help="push the tags in a plan CSV to the Hub")
    sa.add_argument("--plan", required=True)
    sa.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.cmd == "plan":
        plan(a.out)
    else:
        apply(a.plan, a.dry_run)


if __name__ == "__main__":
    main()
