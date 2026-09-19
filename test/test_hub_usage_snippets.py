import pytest

from espnet_model_zoo.hub_usage_snippets import (
    SNIPPET_CLASS,
    already_documented,
    infer_task,
    insert_snippet,
    snippet_for,
    split_front_matter,
)

META_ASR = (
    "files:\n  asr_model_file: exp/a.pth\nyaml_files:\n  asr_train_config: exp/c.yaml\n"
)
META_S2T = (
    "files:\n  s2t_model_file: exp/s2t_train_owsmctc_ebf27/a.pth\n"
    "yaml_files:\n  s2t_train_config: exp/s2t_train_owsmctc_ebf27/config.yaml\n"
)
META_GENERIC = (
    "files:\n  model_file: exp/a.pth\nyaml_files:\n  train_config: exp/c.yaml\n"
)
META_ESPNET3 = (
    "files:\n  model_file: exp/a.pth\n"
    "yaml_files:\n  inference_config: config/inference.yaml\n"
)

FRONT_MATTER = """---
tags:
- espnet
- audio
license: cc-by-4.0
---
"""
CARD = FRONT_MATTER + "\n## ESPnet2 ASR model\n\nThis model was trained by X.\n"


@pytest.mark.parametrize(
    "task, expected_class",
    [
        ("asr", "espnet2.bin.asr_inference.Speech2Text"),
        ("s2t", "espnet2.bin.s2t_inference.Speech2Text"),
        ("s2t_ctc", "espnet2.bin.s2t_inference_ctc.Speech2TextGreedySearch"),
        ("tts", "espnet2.bin.tts_inference.Text2Speech"),
        ("enh", "espnet2.bin.enh_inference.SeparateSpeech"),
        ("spk", "espnet2.bin.spk_inference.Speech2Embedding"),
    ],
)
def test_each_task_names_its_own_class(task, expected_class):
    module, name = expected_class.rsplit(".", 1)
    snippet = snippet_for(task, "espnet/x")
    assert f"{name}.from_pretrained(" in snippet and 'model_tag="espnet/x"' in snippet
    assert SNIPPET_CLASS[task] == expected_class
    # exactly one espnet2 class is imported, and it is this task's
    imports = [ln for ln in snippet.splitlines() if ln.startswith("from espnet2")]
    assert imports == [f"from {module} import {name}"]


def test_the_snippet_names_the_model_and_opens_with_the_heading():
    snippet = snippet_for("tts", "espnet/kan-bayashi_ljspeech_vits")
    assert snippet.startswith("## Usage\n")
    assert snippet.count("espnet/kan-bayashi_ljspeech_vits") == 2  # cli and python


@pytest.mark.parametrize("task", ["s2t_ctc", "tts", "enh"])
def test_the_cli_one_liner_is_shown_for_the_tasks_it_serves(task):
    snippet = snippet_for(task, "espnet/x")
    assert "pip install espnet" in snippet
    assert "\nespnet " in snippet


@pytest.mark.parametrize("task", ["asr", "s2t", "spk"])
def test_no_cli_one_liner_where_the_console_script_cannot_load_the_model(task):
    # `espnet asr` loads Speech2TextGreedySearch, so it serves OWSM-CTC and
    # not the other recognisers; there is no speaker subcommand at all
    assert "pip install espnet" not in snippet_for(task, "espnet/x")


def test_a_task_with_no_checked_snippet_gets_none():
    assert snippet_for("diar", "espnet/x") == ""
    assert snippet_for("", "espnet/x") == ""


def test_meta_keys_name_the_task():
    task, evidence = infer_task("espnet/x", [], ["meta.yaml"], META_ASR)
    assert task == "asr" and "meta.yaml" in evidence


def test_generic_meta_keys_fall_through_to_the_exp_directory():
    task, evidence = infer_task(
        "espnet/x", [], ["exp/enh_train_x/config.yaml", "meta.yaml"], META_GENERIC
    )
    assert task == "enh" and evidence.startswith("file exp/enh_train_x")


def test_an_espnet3_bundle_is_not_given_an_espnet2_snippet():
    task, evidence = infer_task("espnet/x_asr_x", [], ["meta.yaml"], META_ESPNET3)
    assert task == "" and "espnet3" in evidence


def test_owsm_ctc_is_separated_from_the_attention_decoder():
    ctc, _ = infer_task("espnet/owsm_ctc_v4_1B", [], ["meta.yaml"], META_S2T)
    assert ctc == "s2t_ctc"
    attention, _ = infer_task(
        "espnet/owsm_v4_medium_1B",
        [],
        ["exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/config.yaml"],
        None,
    )
    assert attention == "s2t"


def test_a_joint_ctc_attention_asr_recipe_is_not_owsm_ctc():
    # "ctc0.3" is a loss weight in an ordinary ASR recipe, not a CTC model
    task, _ = infer_task(
        "espnet/x_asr_conformer_ctc0.3",
        [],
        ["exp/asr_train_asr_conformer_ctc0.3_raw/config.yaml"],
        None,
    )
    assert task == "asr"


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("espnet/powsm_ctc", "s2t_ctc"),
        ("espnet/owls_1B_180K", "s2t"),
        ("espnet/kan-bayashi_ljspeech_vits", "tts"),
        ("espnet/voxcelebs12_rawnet3", "spk"),
        ("espnet/librispeech_asr_train_asr_conformer", "asr"),
        # the name says enhancement but the model is a recogniser
        ("espnet/x_enh_asr_train_enh_asr_ineube_raw", ""),
        # a speaker count in a TTS name is not the speaker task
        ("espnet/kan-bayashi_vctk_multi_spk_vits", "tts"),
        # nothing in these names decides a class
        ("espnet/parlament", ""),
        ("espnet/cvss-c_en_wavegan_hubert_vocoder", ""),
    ],
)
def test_names_decide_only_what_they_can(model_id, expected):
    task, _ = infer_task(model_id, [], [], None)
    assert task == expected


@pytest.mark.parametrize(
    "path, expected",
    [
        ("exp/asr_train_asr_conformer_raw/config.yaml", "asr"),
        # one model per speaker: the speaker sits between exp/ and the task
        ("exp/a/tts_stats_raw_phn_none/train/feats_stats.npz", "tts"),
        # the directory is not always called exp
        ("save_exp/spk_train_ska_Vox12_emb192_raw_sp/9epoch.pth", "spk"),
        ("exp_owsm/s2t_train_1b_ds_raw_bpe50000/config.yaml", "s2t"),
        # a joint model whose prefix names no single task falls through
        ("exp/diar_enh_train_diar_enh_convtasnet_adapt/config.yaml", ""),
    ],
)
def test_the_exp_directory_is_read_wherever_the_recipe_put_it(path, expected):
    task, _ = infer_task("espnet/x", [], [path], None)
    assert task == expected


def test_a_joint_diarisation_and_enhancement_model_is_not_given_the_enh_class():
    task, evidence = infer_task("espnet/x_librimix_diar_enh_2_3_spk", [], [], None)
    assert task == "" and evidence.startswith("left for review")


def test_files_beat_names():
    task, evidence = infer_task(
        "espnet/x_asr_x", [], ["exp/tts_train_x/config.yaml"], None
    )
    assert task == "tts" and evidence.startswith("file")


def test_split_front_matter():
    front, body = split_front_matter(CARD)
    assert front == FRONT_MATTER
    assert body.startswith("\n## ESPnet2 ASR model")


def test_a_card_without_front_matter_is_all_body():
    assert split_front_matter("# Hello\n") == ("", "# Hello\n")


def test_front_matter_survives_a_rewrite():
    updated = insert_snippet(CARD, snippet_for("asr", "espnet/x"))
    assert updated.startswith(FRONT_MATTER)
    assert split_front_matter(updated)[0] == FRONT_MATTER
    # the snippet comes before the body, and the body is kept whole
    body = split_front_matter(updated)[1]
    assert body.index("## Usage") < body.index("## ESPnet2 ASR model")
    assert "This model was trained by X." in body


def test_a_card_with_no_body_still_gets_the_snippet():
    updated = insert_snippet(FRONT_MATTER, snippet_for("tts", "espnet/x"))
    assert updated == FRONT_MATTER + "\n" + snippet_for("tts", "espnet/x")


@pytest.mark.parametrize(
    "body",
    [
        "\n## Usage\n\nRun it.\n",
        "\n### usage\n\nRun it.\n",
        "\n# Example\n\n```python\nSpeech2Text.from_pretrained('espnet/x')\n```\n",
    ],
)
def test_a_card_that_already_has_usage_is_left_alone(body):
    assert already_documented(FRONT_MATTER + body)


def test_a_plain_recipe_dump_is_not_left_alone():
    assert already_documented(CARD) is None


def test_usage_in_the_front_matter_does_not_count():
    # a "usage" key in the metadata is not a usage section in the card
    assert already_documented("---\nusage: unrestricted\n---\n\n# X\n") is None


def test_a_failed_hub_fetch_leaves_the_row_undecided(monkeypatch):
    from espnet_model_zoo import hub_usage_snippets as hus

    def boom(url, timeout=60):
        raise hus.FetchError(f"{url}: 503")

    monkeypatch.setattr(hus, "_get", boom)
    # the name alone would say ASR; without the files it must not
    task, action, evidence = hus.plan_one("espnet/x_asr_train_asr", [])
    assert task == "" and action == "skip" and evidence.startswith("fetch error")


def test_a_failed_card_fetch_also_leaves_the_row_undecided(monkeypatch):
    from espnet_model_zoo import hub_usage_snippets as hus

    monkeypatch.setattr(
        hus,
        "_get",
        lambda url, timeout=60: {
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "exp/model.pth"}]
        },
    )

    def boom(url, timeout=60):
        raise hus.FetchError(f"{url}: 404")

    monkeypatch.setattr(hus, "_get_text", boom)
    # a card that could not be read might already document usage
    task, action, evidence = hus.plan_one("espnet/x_tts_x", [])
    assert task == "" and action == "skip" and "fetch error" in evidence


def _hub(files, texts):
    def get(url, timeout=60):
        return {"siblings": [{"rfilename": f} for f in files]}

    def get_text(url, timeout=60):
        return texts[url.rsplit("/", 1)[-1]]

    return get, get_text


def test_plan_one_decides_a_card_that_needs_a_snippet(monkeypatch):
    from espnet_model_zoo import hub_usage_snippets as hus

    get, get_text = _hub(
        ["README.md", "meta.yaml", "exp/tts_train_vits/config.yaml"],
        {"meta.yaml": META_GENERIC, "README.md": CARD},
    )
    monkeypatch.setattr(hus, "_get", get)
    monkeypatch.setattr(hus, "_get_text", get_text)
    assert hus.plan_one("espnet/x", []) == (
        "tts",
        "add",
        "file exp/tts_train_vits/config.yaml",
    )


def test_plan_one_skips_a_card_stored_through_git_lfs(monkeypatch):
    from espnet_model_zoo import hub_usage_snippets as hus

    pointer = (
        "version https://git-lfs.github.com/spec/v1\n" "oid sha256:0000\nsize 12\n"
    )
    get, get_text = _hub(
        ["README.md", "meta.yaml"], {"meta.yaml": META_ASR, "README.md": pointer}
    )
    monkeypatch.setattr(hus, "_get", get)
    monkeypatch.setattr(hus, "_get_text", get_text)
    task, action, evidence = hus.plan_one("espnet/x", [])
    assert action == "skip" and "Git LFS" in evidence


def test_plan_one_skips_an_empty_repository(monkeypatch):
    from espnet_model_zoo import hub_usage_snippets as hus

    get, _ = _hub([".gitattributes", "README.md"], {})
    monkeypatch.setattr(hus, "_get", get)
    monkeypatch.setattr(hus, "_get_text", lambda url, timeout=60: CARD)
    assert hus.plan_one("espnet/x_asr", [])[1:] == (
        "skip",
        "empty repository",
    )


def test_plan_one_reports_a_task_that_has_no_snippet(monkeypatch):
    from espnet_model_zoo import hub_usage_snippets as hus

    get, get_text = _hub(
        ["README.md", "exp/diar_train_x/config.yaml"], {"README.md": CARD}
    )
    monkeypatch.setattr(hus, "_get", get)
    monkeypatch.setattr(hus, "_get_text", get_text)
    task, action, evidence = hus.plan_one("espnet/x", [])
    assert task == "diar" and action == "skip" and "no checked snippet" in evidence
