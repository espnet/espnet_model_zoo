#!/usr/bin/env python3
"""Put a runnable usage snippet at the top of the espnet Hub model cards.

    python -m espnet_model_zoo.hub_usage_snippets plan  [--out usage_snippets.csv]
    python -m espnet_model_zoo.hub_usage_snippets apply --plan usage_snippets.csv \
[--dry-run]

`plan` reads every model under the organisation and works out which espnet2
inference class loads it, from the same evidence `hub_pipeline_tags` uses and
in the same order of trust: the constructor keys in its meta.yaml, then the
task prefix of its exp/ directory, then words in its name. It writes one row
per model - the task, whether the card would be edited, and the evidence -
so the CSV can be read and edited before `apply` rewrites the cards. A model
whose task nothing decides, or whose task has no snippet that was checked
against released espnet, gets no snippet and a row saying why.

`apply` inserts the snippet between the card's front matter and its body,
under a `## Usage` heading, and pushes it. It never writes twice: a card that
already has a usage section, or that already calls `from_pretrained`, is left
alone. The front matter is carried over byte for byte - the card is edited as
text rather than re-serialised - and is checked to be unchanged before the
push. `apply` needs a token with write access to the organisation (`hf auth
login`); `plan` needs no token.

A card that is a recipe dump with no example is the organisation's normal
case: of the 643 cards on the Hub on 2026-09-19, 614 had neither a usage
section nor a `from_pretrained` call anywhere in them.
"""

import argparse
import csv
import re
import sys
from typing import Dict, List, Optional, Tuple

from espnet_model_zoo.hub_pipeline_tags import (
    API,
    ORG,
    TASK_TAG,
    FetchError,
    _get,
    _get_text,
    has_model_files,
    strip_markdown_lfs_rules,
)

# espnet task -> the espnet2 class whose ``from_pretrained`` loads that task's
# models. The pairing is not a matter of taste: ``from_pretrained`` hands the
# downloader's keys straight to the constructor, so an ASR model - which packs
# ``asr_train_config`` - only fits asr_inference.Speech2Text, and an OWSM
# model, which packs ``s2t_train_config``, only fits the s2t classes.
SNIPPET_CLASS = {
    "asr": "espnet2.bin.asr_inference.Speech2Text",
    "s2t": "espnet2.bin.s2t_inference.Speech2Text",
    "s2t_ctc": "espnet2.bin.s2t_inference_ctc.Speech2TextGreedySearch",
    "tts": "espnet2.bin.tts_inference.Text2Speech",
    "enh": "espnet2.bin.enh_inference.SeparateSpeech",
    "spk": "espnet2.bin.spk_inference.Speech2Embedding",
}

# The `espnet` console script of espnet 202610, for the tasks it covers.
# `espnet asr` loads Speech2TextGreedySearch and nothing else, so it serves
# OWSM-CTC and not the other recognisers; there is no speaker subcommand.
# `espnet models` names the default model of each subcommand.
CLI_COMMAND = {
    "s2t_ctc": "espnet asr audio.wav --model {tag} --language eng",
    "tts": 'espnet tts "Hello from ESPnet" -o out.wav --model {tag}',
    "enh": "espnet enhance noisy.wav -o enhanced.wav --model {tag}",
}

PYTHON_SNIPPET = {
    "asr": """import soundfile as sf
from espnet2.bin.asr_inference import Speech2Text

speech2text = Speech2Text.from_pretrained(model_tag="{tag}")
speech, rate = sf.read("audio.wav")  # a single channel, at the model's rate
text, *_ = speech2text(speech)[0]
print(text)""",
    # The attention decoder is driven by the language and task symbols; both
    # can also be given per utterance, as s2t(speech, lang_sym=..., ...).
    "s2t": """import soundfile as sf
from espnet2.bin.s2t_inference import Speech2Text

s2t = Speech2Text.from_pretrained(
    model_tag="{tag}", lang_sym="<eng>", task_sym="<asr>", beam_size=5
)
speech, rate = sf.read("audio.wav")  # 16 kHz; padded or trimmed to 30 s
text, *_ = s2t(speech)[0]
print(text)""",
    # batch_decode takes the audio path itself and handles long-form audio by
    # chunking it, which is what makes OWSM-CTC worth using from Python.
    "s2t_ctc": """from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

s2t = Speech2TextGreedySearch.from_pretrained(model_tag="{tag}")
# <eng>, <jpn>, ... ; OWSM models also accept <nolang> to detect the language
print(s2t.batch_decode("audio.wav", lang_sym="<eng>", task_sym="<asr>"))""",
    "tts": """import soundfile as sf
from espnet2.bin.tts_inference import Text2Speech

tts = Text2Speech.from_pretrained(model_tag="{tag}")
output = tts("Hello from ESPnet")
sf.write("out.wav", output["wav"].view(-1).cpu().numpy(), tts.fs)""",
    # SeparateSpeech takes (Batch, Nsamples [, Channels]) and returns one wave
    # per output stream: one for enhancement, one per speaker for separation.
    "enh": """import soundfile as sf
from espnet2.bin.enh_inference import SeparateSpeech

enh = SeparateSpeech.from_pretrained(model_tag="{tag}")
speech, rate = sf.read("noisy.wav", dtype="float32")
waves = enh(speech[None, ...], fs=rate)
sf.write("enhanced.wav", waves[0][0], rate)""",
    "spk": """import soundfile as sf
from espnet2.bin.spk_inference import Speech2Embedding

speech2embedding = Speech2Embedding.from_pretrained(model_tag="{tag}")
speech, rate = sf.read("audio.wav")  # 16 kHz
embedding = speech2embedding(speech)  # (1, embedding_dim)""",
}

# espnet3's pack_model writes these instead of the <task>_train_config /
# <task>_model_file pairs the espnet2 classes take; espnet2.utils.pretrained
# refuses such a bundle outright, so no espnet2 snippet can be right for one.
_ESPNET3_KEYS = re.compile(r"^\s+(inference_config|training_config):", re.M)

# Only the recognisers whose exp/ directory or meta.yaml key is "s2t" can be
# OWSM-CTC. Within that set this agreed with the `model: espnet_ctc` line of
# every s2t model's own config.yaml on the Hub on 2026-09-19 (26 checked; the
# 27th, owsm_v1, predates that key). It is deliberately narrow: "ctc" alone
# would catch every joint CTC/attention recipe, which trains an ASR model.
_OWSM_CTC = re.compile(r"owsm[-_]?ctc|multitask-ctc|owsmctc|(^|[-_])ctc([-_]|$)", re.I)

# Words in a name that name an espnet task on their own, for the models that
# ship neither a meta.yaml nor an exp/ directory. First match wins, and the
# order is defensive: enh_asr is a recogniser, not enhancement, and the "spk"
# in a multi-speaker TTS name is a speaker count rather than the speaker task.
# A word that names only a Hub pipeline (a vocoder, an SSL encoder, "asr" on a
# model that may be OWSM) cannot name a class, so it is left undecided.
NAME_TASKS: List[Tuple[str, str]] = [
    (r"(^|[\s_/-])enh_(asr|s2t|st)([\s_/-]|$)", ""),
    # diar_enh is a joint diarisation model, which SeparateSpeech will not load
    (r"(^|[\s_/-])(diar|vad)([\s_/-]|$)|diariz", ""),
    (r"owsm[-_]?ctc", "s2t_ctc"),  # also catches powsm_ctc
    (r"(^|[\s_/-])(owsm|owls|powsm)", "s2t"),
    (r"(^|[\s_/-])(vits|jets|fastspeech|tacotron|tts)([\s_/-]|$)", "tts"),
    (r"(^|[\s_/-])(enh|tse)([\s_/-]|$)", "enh"),
    (r"(^|[\s_/-])spk([\s_/-]|$)|voxceleb|rawnet|ecapa", "spk"),
    (r"(^|[\s_/-])asr([\s_/-]|$)", "asr"),
]


def task_from_meta(meta_yaml: Optional[str]) -> Optional[Tuple[str, str]]:
    """The espnet task named by the constructor keys pack.py wrote."""
    if not meta_yaml:
        return None
    if _ESPNET3_KEYS.search(meta_yaml):
        return "", "packed by espnet3; espnet2's from_pretrained refuses it"
    for key in re.findall(
        r"^\s+([a-z_0-9]+?)_(?:train_config|model_file):", meta_yaml, re.M
    ):
        task = key.replace("_train", "")
        if key in ("train", "model"):
            continue  # tts/enh/spk pack without a task in the key
        if task in TASK_TAG:
            return task, f"meta.yaml key {key}_*"
    return None


def task_from_files(files: List[str]) -> Optional[Tuple[str, str]]:
    """exp/<task>_train_.../ and exp/<task>_stats_... carry the task prefix.

    Neither half of that path is fixed. The directory is ``exp_owsm`` and
    ``exp_360`` for OWLS and ``save_exp`` for the voxceleb speaker models,
    and a recipe with one model per speaker puts the speaker in between
    (``exp/a/tts_train_tacotron2_raw_phn_none``). Insisting on a bare ``exp/``
    left 26 talromur voices and 14 speaker models with no evidence but their
    names, which for "tacotron2" and "fastspeech2" is none at all.
    """
    for f in files:
        m = re.match(
            r"[a-z_0-9]*exp[a-z_0-9]*/(?:[a-z_0-9]+/)?([a-z_0-9]+?)_(?:train|stats)", f
        )
        if m and m.group(1) in TASK_TAG:
            return m.group(1), f"file {f}"
    return None


def task_from_name(model_id: str, tags: List[str]) -> Optional[Tuple[str, str]]:
    text = (model_id.split("/", 1)[-1] + " " + " ".join(tags)).lower()
    for pattern, task in NAME_TASKS:
        m = re.search(pattern, text)
        if m:
            if not task:
                return "", f"left for review: name matches {m.group(0)!r}"
            return task, f"name/tags match {m.group(0)!r}"
    return None


def refine_s2t(task: str, model_id: str, files: List[str]) -> str:
    """Split the s2t models into the CTC and the attention decoder.

    They are two different classes with two different calls, and they pack
    identical keys, so the recipe name in the exp/ directory - or the model's
    own name, for the ones packed from a directory that does not carry it -
    is the only evidence short of downloading a 50000-line config.
    """
    if task != "s2t":
        return task
    names = [f.rsplit("/", 1)[0] for f in files if f.startswith("exp")]
    names.append(model_id.split("/", 1)[-1])
    return "s2t_ctc" if any(_OWSM_CTC.search(n) for n in names) else "s2t"


def infer_task(
    model_id: str, tags: List[str], files: List[str], meta_yaml: Optional[str]
) -> Tuple[str, str]:
    """Return the espnet task of a model and the evidence for it."""
    for source in (
        lambda: task_from_meta(meta_yaml),
        lambda: task_from_files(files),
        lambda: task_from_name(model_id, tags),
    ):
        hit = source()
        if hit:
            task, evidence = hit
            return refine_s2t(task, model_id, files), evidence
    return "", "no evidence"


def snippet_for(task: str, model_id: str) -> str:
    """The markdown to insert, or "" for a task with no checked snippet."""
    if task not in PYTHON_SNIPPET:
        return ""
    parts = ["## Usage", ""]
    command = CLI_COMMAND.get(task)
    if command:
        parts += ["```bash", "pip install espnet", command.format(tag=model_id), "```"]
        parts.append("")
    parts += ["```python", PYTHON_SNIPPET[task].format(tag=model_id), "```"]
    return "\n".join(parts) + "\n"


_FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n.*?\r?\n---[ \t]*\r?\n", re.S)


def split_front_matter(card: str) -> Tuple[str, str]:
    """Split a card into its front matter, delimiters included, and its body."""
    m = _FRONT_MATTER.match(card)
    if not m:
        return "", card
    return m.group(0), card[m.end() :]


_USAGE_HEADING = re.compile(r"^#{1,6}[ \t]*usage\b.*$", re.M | re.I)


def already_documented(card: str) -> Optional[str]:
    """Why this card must be left alone, or None if it may be edited."""
    body = split_front_matter(card)[1]
    heading = _USAGE_HEADING.search(body)
    if heading:
        return f"card already has a {heading.group(0).strip()!r} section"
    if "from_pretrained" in body:
        return "card already calls from_pretrained"
    return None


def insert_snippet(card: str, snippet: str) -> str:
    """Put ``snippet`` between the card's front matter and its body.

    The front matter is copied, never parsed and written back: re-serialising
    it reorders keys, requotes values and drops comments, which would show up
    as a metadata change on every model the tool touches.
    """
    front, body = split_front_matter(card)
    body = body.lstrip("\n")
    out = front + ("\n" if front else "") + snippet.rstrip("\n") + "\n"
    if body:
        out += "\n" + body
    return out


# What the raw endpoint returns for a file stored through Git LFS. The card's
# real text is behind the pointer, so nothing can be decided from this.
_LFS_POINTER = "version https://git-lfs.github.com/spec/v1"


def plan_one(model_id: str, tags: List[str]) -> Tuple[str, str, str]:
    """Return (task, action, evidence) for one model.

    A request that fails never becomes "no evidence": the name rules would
    then decide a model whose own files might have said otherwise, and
    `apply` would push a snippet built on that.
    """
    try:
        info = _get(f"{API}/models/{model_id}")
        files = [s["rfilename"] for s in info.get("siblings", [])]
        meta = None
        if "meta.yaml" in files:
            meta = _get_text(f"https://huggingface.co/{model_id}/raw/main/meta.yaml")
        card = ""
        if "README.md" in files:
            card = _get_text(f"https://huggingface.co/{model_id}/raw/main/README.md")
    except FetchError as e:
        return "", "skip", f"fetch error, not decided: {e}"
    if not has_model_files(files):
        return "", "skip", "empty repository"
    if card.startswith(_LFS_POINTER):
        # the body is invisible, so "does it already document usage?" is
        # unanswerable; hub_pipeline_tags fix-cards takes the card out of LFS
        return "", "skip", "card stored through Git LFS; run fix-cards first"
    task, evidence = infer_task(model_id, tags, files, meta)
    if not task:
        return "", "skip", evidence
    if not snippet_for(task, model_id):
        return task, "skip", f"no checked snippet for task {task!r} ({evidence})"
    documented = already_documented(card)
    if documented:
        return task, "skip", f"{documented} ({evidence})"
    return task, "add", evidence


_FIELDS = ["model", "task", "action", "evidence", "downloads"]


def plan(out_path: str) -> None:
    models = _get(f"{API}/models?author={ORG}&limit=1000")  # FetchError aborts
    print(f"{len(models)} models", file=sys.stderr)
    rows: List[Dict[str, str]] = []
    for i, m in enumerate(models, 1):
        mid = m["modelId"]
        task, action, evidence = plan_one(mid, m.get("tags", []))
        rows.append(
            {
                "model": mid,
                "task": task,
                "action": action,
                "evidence": evidence,
                "downloads": str(m.get("downloads", "")),
            }
        )
        if i % 50 == 0:
            print(f"  {i}/{len(models)}", file=sys.stderr)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
    adding = [r for r in rows if r["action"] == "add"]
    by_task: Dict[str, int] = {}
    for r in adding:
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
    print(f"wrote {out_path}: {len(adding)} cards would get a snippet", file=sys.stderr)
    for task, n in sorted(by_task.items(), key=lambda kv: -kv[1]):
        print(f"  {task:8} {n:4}  {SNIPPET_CLASS[task]}", file=sys.stderr)
    print(f"  {len(rows) - len(adding)} skipped", file=sys.stderr)


def _repair_gitattributes(api, model_id: str, dry_run: bool) -> None:
    """Stop the card we are about to write from being swallowed by Git LFS.

    A ``*.md`` LFS rule makes the pushed card a pointer file, which the Hub's
    parser cannot read: the model would lose its metadata as the price of
    gaining an example. The rule has to be committed before the card, because
    the Hub decides a file's storage from the .gitattributes already in the
    repository rather than from one in the same commit.
    """
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.errors import EntryNotFoundError

    try:
        local = api.hf_hub_download(repo_id=model_id, filename=".gitattributes")
    except EntryNotFoundError:
        return
    with open(local, encoding="utf-8") as f:
        fixed, removed = strip_markdown_lfs_rules(f.read())
    if not removed:
        return
    print(f"{model_id}: dropping {', '.join(removed)} from .gitattributes")
    if dry_run:
        return
    api.create_commit(
        repo_id=model_id,
        repo_type="model",
        operations=[CommitOperationAdd(".gitattributes", fixed.encode("utf-8"))],
        commit_message="Stop sending markdown through Git LFS",
    )


def apply(plan_path: str, dry_run: bool) -> int:
    """Push the snippets in a plan; returns the number of models that failed.

    Every model is tried and every failure is named in the summary. The card
    is read again here rather than taken from the plan, so a card that gained
    a usage section since the plan was written is still left alone.
    """
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import EntryNotFoundError

    with open(plan_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["action"] == "add"]
    print(
        f"{len(rows)} cards to edit" + (" (dry run)" if dry_run else ""),
        file=sys.stderr,
    )
    api = HfApi()
    failed: List[Tuple[str, str]] = []
    for r in rows:
        mid, task = r["model"], r["task"]
        try:
            snippet = snippet_for(task, mid)
            if not snippet:
                raise ValueError(f"no snippet for task {task!r}")
            try:
                local = api.hf_hub_download(repo_id=mid, filename="README.md")
                with open(local, encoding="utf-8") as f:
                    card = f.read()
            except EntryNotFoundError:
                card = ""  # the push creates a card holding only the snippet
            documented = already_documented(card)
            if documented:
                print(f"{mid}: {documented}, skipped")
                continue
            updated = insert_snippet(card, snippet)
            if split_front_matter(updated)[0] != split_front_matter(card)[0]:
                raise AssertionError("front matter changed")
            print(f"{mid}: {task} snippet, {SNIPPET_CLASS[task]}   [{r['evidence']}]")
            _repair_gitattributes(api, mid, dry_run)
            if dry_run:
                continue
            api.create_commit(
                repo_id=mid,
                repo_type="model",
                operations=[CommitOperationAdd("README.md", updated.encode("utf-8"))],
                commit_message="Add a usage example to the model card",
            )
        except Exception as e:  # keep going; the summary names every failure
            first = (str(e).strip().splitlines() or [type(e).__name__])[0]
            print(f"{mid}: FAILED {type(e).__name__}: {first}", file=sys.stderr)
            failed.append((mid, first))
    if failed:
        print(f"\n{len(failed)} of {len(rows)} models failed:", file=sys.stderr)
        for mid, why in failed:
            print(f"  {mid}: {why}", file=sys.stderr)
    return len(failed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("plan", help="work out each model's snippet; write a CSV")
    sp.add_argument("--out", default="usage_snippets.csv")
    sa = sub.add_parser("apply", help="push the snippets in a plan CSV to the Hub")
    sa.add_argument("--plan", required=True)
    sa.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.cmd == "plan":
        plan(a.out)
    else:
        sys.exit(1 if apply(a.plan, a.dry_run) else 0)


if __name__ == "__main__":
    main()
