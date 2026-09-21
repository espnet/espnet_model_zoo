# Maintaining the espnet organization on Hugging Face

Notes for administrators of [huggingface.co/espnet](https://huggingface.co/espnet). Nothing
here is needed to *use* a model — see the [README](README.md) for that.

The organization holds 671 models as of 2026-09-21. Most were pushed by a recipe years
apart, under conventions that changed while they were being pushed, so a share of them are
missing metadata that the Hub or `ModelDownloader` needs. Three tools find those and repair
them in bulk.

## The shape they share

Each tool has two steps:

| | Needs a token | Writes |
| :-- | :-- | :-- |
| `plan` | no | nothing — a CSV you read and edit |
| `apply` | yes, write access to the organization (`hf auth login`) | one small file per model |

`plan` writes one row per model with the evidence behind its decision, and leaves the
decision empty when nothing settles it. Edit the CSV before applying it.

No weights are ever re-uploaded, which is what makes running one of these across the whole
organization cheap. What `apply` writes differs: `hub_meta_yaml` adds a `meta.yaml`,
`hub_pipeline_tags` rewrites a card's front matter, and `hub_usage_snippets` edits the
card's body — that one commits model-card content, not just metadata.

`--dry-run` also differs, and only one of the three validates:

| | `--dry-run` does |
| :-- | :-- |
| `hub_meta_yaml` | prints the `meta.yaml` it would upload |
| `hub_pipeline_tags` | runs the Hub's validator on the changed card |
| `hub_usage_snippets` | prints the edit and stops before the commit; no validation |

Each tool prints the models that failed while it was working and exits non-zero if any
did — read that list rather than the exit code alone. Rows a `plan` marked as blocked are
a separate matter: `apply` never attempts them, so they appear in neither list. They stay
in the plan CSV, which is the only place they are recorded.

## `hub_meta_yaml` — models that will not load

```sh
python -m espnet_model_zoo.hub_meta_yaml plan --out meta_plan.csv
python -m espnet_model_zoo.hub_meta_yaml apply --plan meta_plan.csv --dry-run
python -m espnet_model_zoo.hub_meta_yaml apply --plan meta_plan.csv
```

A model published with `espnet2.bin.pack` carries `meta.yaml` at its root, naming the
training config and the checkpoint under the key names the task's inference class takes;
`ModelDownloader` hands those straight to the constructor. A repository uploaded by hand
holds the same files with nothing saying which is which, so `download_and_unpack` refuses
it with a `RuntimeError` listing the repository's files and asking the caller to pass
`train_config` and `model_file` themselves.

`plan` works out which file is the config, which is the checkpoint, and what the task is.
`apply` uploads one small text file per repository.

What `meta.yaml` cannot fix is named in the **`blockers`** column: a config referring to a
bpemodel or a normalisation statistics file the repository does not contain, or a
checkpoint that is a dangling symlink. Those need the missing file, not this script.
`apply` skips every row with a non-empty `blockers` and says nothing about them, so the
CSV is where you find them.

Nor does `meta.yaml` make an old model build against current espnet — a config written for
a since-removed argument still fails, and the plan says nothing about that. Load the model
to find out.

## `hub_pipeline_tags` — models the Hub cannot filter

```sh
python -m espnet_model_zoo.hub_pipeline_tags plan --out plan.csv
python -m espnet_model_zoo.hub_pipeline_tags apply --plan plan.csv --dry-run
python -m espnet_model_zoo.hub_pipeline_tags apply --plan plan.csv
```

A model without a `pipeline_tag` does not appear when the Hub is filtered by task, and gets
no task widget. 268 of 667 models were in that state on 2026-09-17; 47 of 671 remained on
2026-09-21.

`plan` infers the tag from, in order of trust:

1. the constructor keys in `meta.yaml` (`asr_train_config` → ASR),
2. the task prefix of the `exp/` directory,
3. words in the model's name and tags.

`apply` rewrites each card's front matter: it sets the tag, leaves a tag someone set by
hand alone, and repairs the language codes the Hub's validator rejects.

## `hub_usage_snippets` — cards with no example

```sh
python -m espnet_model_zoo.hub_usage_snippets plan --out usage_snippets.csv
python -m espnet_model_zoo.hub_usage_snippets apply --plan usage_snippets.csv --dry-run
python -m espnet_model_zoo.hub_usage_snippets apply --plan usage_snippets.csv
```

A card that is a recipe dump with no example is the organization's normal state. This works
out which `espnet2` inference class loads the model — same evidence and same order of trust
as `hub_pipeline_tags` — and inserts a runnable snippet between the front matter and the
body, under a `## Usage` heading.

It never writes twice: a card that already has a usage section, or that already calls
`from_pretrained`, is left alone. The card is edited as text rather than re-serialised, so
the front matter is carried over rather than regenerated, and the tool refuses to push if
it changed. That check compares the two sides with trailing whitespace stripped, so it
would not catch a difference in trailing whitespace alone. A model whose task nothing decides, or whose task has no snippet that was checked
against released espnet, gets no snippet and a row saying why.

## Releasing

Registering a model means a row in [table.csv](espnet_model_zoo/table.csv), which puts it
under CI. After merging such a pull request, increment the third version number in
[setup.py](setup.py) and push the `v*` tag to release.
