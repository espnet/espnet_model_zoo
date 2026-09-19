from email.utils import format_datetime

import pytest

from espnet_model_zoo.hub_pipeline_tags import (
    fix_language,
    has_model_files,
    hub_reads_card,
    infer,
    strip_markdown_lfs_rules,
    update_card_data,
)

META_ASR = (
    "files:\n  asr_model_file: exp/a.pth\nyaml_files:\n  asr_train_config: exp/c.yaml\n"
)
META_GENERIC = (
    "files:\n  model_file: exp/a.pth\nyaml_files:\n  train_config: exp/c.yaml\n"
)


def test_meta_keys_name_the_task():
    tag, evidence = infer("espnet/x", [], ["meta.yaml"], META_ASR)
    assert tag == "automatic-speech-recognition" and "meta.yaml" in evidence


def test_generic_meta_keys_fall_through_to_the_exp_directory():
    # tts, enh and spk pack as train_config/model_file, which names no task
    tag, evidence = infer(
        "espnet/x", [], ["exp/enh_train_x/config.yaml", "meta.yaml"], META_GENERIC
    )
    assert tag == "audio-to-audio" and evidence.startswith("file exp/enh_train_x")


def test_exp_stats_directory_counts_too():
    tag, _ = infer(
        "espnet/x", [], ["exp/svs_stats_raw_phn/train/feats_stats.npz"], None
    )
    assert tag == "text-to-speech"


@pytest.mark.parametrize(
    "model_id, expected",
    [
        # task words in the name
        ("espnet/librispeech_asr_train_asr_conformer", "automatic-speech-recognition"),
        ("espnet/owls_1b_22K", "automatic-speech-recognition"),
        ("espnet/speechlm_tts_v1", "text-to-speech"),
        ("espnet/BSCodec", "audio-to-audio"),
        ("espnet/ms_snsd_tfgridnet", "audio-to-audio"),
        ("espnet/brianyan918_must_c_v2_en-de_st_multidecoder", "translation"),
        ("espnet/cvss_s2st_discrete_unit", "audio-to-audio"),  # speech in, speech out
        ("espnet/voxcelebs12_rawnet3", "audio-classification"),
        ("espnet/geolid_vl107only_shared_frozen", "audio-classification"),
        ("espnet/meld_cls1_wavlm_base_plus", "audio-classification"),
        ("espnet/diar_ami_eend_eda", "voice-activity-detection"),
        ("espnet/x_enh_asr_train_enh_asr_ineube_raw", "automatic-speech-recognition"),
        # the first version of the rules got each of these wrong
        (
            "espnet/kan-bayashi_vctk_multi_spk_vits",
            "text-to-speech",
        ),  # spk is a speaker count
        (
            "espnet/cvss-c_en_wavegan_hubert_vocoder",
            "audio-to-audio",
        ),  # a vocoder, not ASR
        ("espnet/WavLabLM-MS-40k", "feature-extraction"),  # a bare SSL encoder
        ("espnet/hubert_dummy", "feature-extraction"),
        # nothing decides these: left for a person
        (
            "espnet/bur_openslr80_hubert",
            "",
        ),  # fine-tune or encoder? the name cannot say
        ("espnet/universa-wavlm_base_urgent24_multi-metric", ""),  # a quality predictor
        ("espnet/owls_1B_180K_intermediates", ""),
        ("espnet/parlament", ""),
    ],
)
def test_names(model_id, expected):
    tag, _ = infer(model_id, [], [], None)
    assert tag == expected


def test_files_beat_names():
    # the exp/ directory is the model's own word; the name is the uploader's
    tag, evidence = infer("espnet/x_asr_x", [], ["exp/tts_train_x/config.yaml"], None)
    assert tag == "text-to-speech" and evidence.startswith("file")


def test_rate_limiting_is_waited_out_rather_than_reported_as_a_failure(monkeypatch):
    from espnet_model_zoo import hub_pipeline_tags as hpt

    class _Response:
        def __init__(self, status_code):
            self.status_code = status_code
            self.headers = {"Retry-After": "0"}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise hpt.requests.HTTPError(str(self.status_code))

        def json(self):
            return {"ok": True}

    answers = [_Response(429), _Response(503), _Response(200)]
    monkeypatch.setattr(hpt.requests, "get", lambda url, timeout: answers.pop(0))
    monkeypatch.setattr(hpt.time, "sleep", lambda seconds: None)
    assert hpt._get("https://example/x") == {"ok": True}
    assert not answers


@pytest.mark.parametrize(
    "header, expected",
    [
        ("30", 30.0),
        (None, 4.0),  # no header: the exponential backoff for attempt 2
        ("not a number", 4.0),
        ("nan", 4.0),
        ("-5", 0.0),  # invalid, and it must not reach time.sleep
        ("Mon, 21 Oct 2013 07:28:00 GMT", 0.0),  # an expired date means now
        ("86400", 60.0),  # a server asking for a day must not stall the sweep
    ],
)
def test_retry_after_is_parsed_and_bounded(header, expected):
    from espnet_model_zoo import hub_pipeline_tags as hpt

    class _Response:
        headers = {} if header is None else {"Retry-After": header}

    assert hpt.retry_delay(_Response(), 2) == expected


def test_retry_after_may_be_an_http_date_in_the_future():
    from datetime import datetime, timedelta, timezone

    from espnet_model_zoo import hub_pipeline_tags as hpt

    soon = datetime.now(timezone.utc) + timedelta(seconds=10)

    class _Response:
        headers = {"Retry-After": format_datetime(soon, usegmt=True)}

    assert 5.0 < hpt.retry_delay(_Response(), 0) <= 10.0


def test_rate_limiting_that_never_lets_up_is_a_fetch_error(monkeypatch):
    from espnet_model_zoo import hub_pipeline_tags as hpt

    class _Limited:
        status_code = 429
        headers: dict = {}

        def raise_for_status(self):
            raise hpt.requests.HTTPError("429")

    monkeypatch.setattr(hpt.requests, "get", lambda url, timeout: _Limited())
    monkeypatch.setattr(hpt.time, "sleep", lambda seconds: None)
    with pytest.raises(hpt.FetchError):
        hpt._get("https://example/x")


def test_a_failed_hub_fetch_leaves_the_row_undecided(monkeypatch):
    from espnet_model_zoo import hub_pipeline_tags as hpt

    def boom(url, timeout=60):
        raise hpt.FetchError(f"{url}: 503")

    monkeypatch.setattr(hpt, "_get", boom)
    # the name alone would say ASR; without the files it must not
    tag, evidence = hpt.plan_one("espnet/x_asr_train_asr", [])
    assert tag == "" and evidence.startswith("fetch error")


def test_a_failed_meta_fetch_also_leaves_the_row_undecided(monkeypatch):
    from espnet_model_zoo import hub_pipeline_tags as hpt

    monkeypatch.setattr(
        hpt, "_get", lambda url, timeout=60: {"siblings": [{"rfilename": "meta.yaml"}]}
    )

    def boom(url, timeout=60):
        raise hpt.FetchError(f"{url}: 404")

    monkeypatch.setattr(hpt, "_get_text", boom)
    tag, evidence = hpt.plan_one("espnet/x_tts_x", [])
    assert tag == "" and "fetch error" in evidence


@pytest.mark.parametrize(
    "value, expected",
    [
        ("noinfo", None),
        ("jp", "ja"),
        ("en", "en"),
        (["en", "noinfo"], ["en"]),
        (["jp", "ja"], ["ja"]),
        (["noinfo"], None),
        (None, None),
    ],
)
def test_fix_language(value, expected):
    assert fix_language(value) == expected


def test_update_card_data_adds_the_tag_and_repairs_language():
    data = {"tags": ["espnet"], "language": "noinfo"}
    notes = update_card_data(data, "voice-activity-detection")
    assert data == {"tags": ["espnet"], "pipeline_tag": "voice-activity-detection"}
    assert len(notes) == 2


def test_update_card_data_keeps_a_tag_set_by_hand():
    data = {"pipeline_tag": "audio-classification", "language": "en"}
    assert update_card_data(data, "automatic-speech-recognition") == []
    assert data["pipeline_tag"] == "audio-classification"


def test_update_card_data_leaves_a_valid_language_alone():
    data = {"language": ["en", "de"]}
    assert update_card_data(data, "translation") == ["pipeline_tag: translation"]
    assert data["language"] == ["en", "de"]


def test_has_model_files_ignores_git_metadata_and_the_card():
    assert not has_model_files([".gitattributes"])
    assert not has_model_files([".gitattributes", "README.md"])
    assert has_model_files([".gitattributes", "exp/model.pth"])


GITATTRIBUTES = (
    "*.bin filter=lfs diff=lfs merge=lfs -text\n"
    "*.md filter=lfs diff=lfs merge=lfs -text\n"
    "*.ark filter=lfs diff=lfs merge=lfs -text\n"
)


def test_strip_markdown_lfs_rules_keeps_the_other_rules():
    fixed, removed = strip_markdown_lfs_rules(GITATTRIBUTES)
    assert removed == ["*.md filter=lfs diff=lfs merge=lfs -text"]
    assert fixed == (
        "*.bin filter=lfs diff=lfs merge=lfs -text\n"
        "*.ark filter=lfs diff=lfs merge=lfs -text\n"
    )


def test_strip_markdown_lfs_rules_also_matches_the_readme_by_name():
    fixed, removed = strip_markdown_lfs_rules(
        "README.md filter=lfs diff=lfs merge=lfs -text\n*.pth filter=lfs -text\n"
    )
    assert removed == ["README.md filter=lfs diff=lfs merge=lfs -text"]
    assert fixed == "*.pth filter=lfs -text\n"


def test_strip_markdown_lfs_rules_leaves_a_clean_file_alone():
    clean = "*.bin filter=lfs diff=lfs merge=lfs -text\n"
    assert strip_markdown_lfs_rules(clean) == (clean, [])
    # a plain text rule for markdown is not an LFS rule
    text_rule = "*.md text\n"
    assert strip_markdown_lfs_rules(text_rule) == (text_rule, [])


class _Info:
    def __init__(self, card_data):
        self.card_data = card_data


def test_hub_reads_card():
    assert not hub_reads_card(_Info(None))
    assert not hub_reads_card(_Info({}))
    assert hub_reads_card(_Info({"pipeline_tag": "text-to-speech"}))


def test_hub_reads_card_accepts_a_card_data_object():
    class Data:
        def to_dict(self):
            return {"language": "en"}

    assert hub_reads_card(_Info(Data()))


def test_a_language_that_says_unknown_is_dropped_not_guessed():
    from espnet_model_zoo.hub_pipeline_tags import drop_unpublishable_metadata

    data = {"tags": ["espnet"], "language": "noinfo", "license": "cc-by-4.0"}
    errors = ['Error: "language" with value "noinfo" is not valid.']
    out, notes = drop_unpublishable_metadata(dict(data), errors)
    assert "language" not in out
    assert out["tags"] == ["espnet"] and out["license"] == "cc-by-4.0"
    assert notes == ["dropped language: 'noinfo'"]


def test_a_key_with_no_value_needs_no_help_from_the_hub():
    from espnet_model_zoo.hub_pipeline_tags import drop_unpublishable_metadata

    # the Hub's own validator misses this one: loading a card drops the key
    # before anything is checked, so it is found locally and with no errors
    out, notes = drop_unpublishable_metadata({"language": None, "license": "mit"}, [])
    assert out == {"license": "mit"}
    assert notes == ["dropped empty language"]


def test_one_bad_dataset_does_not_take_the_good_ones_with_it():
    from espnet_model_zoo.hub_pipeline_tags import drop_unpublishable_metadata

    data = {"datasets": ["librispeech_asr", "librispeech 960h"]}
    errors = ['Error: "datasets[1]" with value "librispeech 960h" is not valid.']
    out, notes = drop_unpublishable_metadata(data, errors)
    assert out == {"datasets": ["librispeech_asr"]}
    assert notes == ["dropped datasets entry 'librispeech 960h'"]


def test_a_list_left_empty_goes_too():
    from espnet_model_zoo.hub_pipeline_tags import drop_unpublishable_metadata

    data = {"datasets": ["accented french (openslr56)"], "license": "cc-by-4.0"}
    errors = [
        'Error: "datasets[0]" with value "accented french (openslr56)" is not valid.'
    ]
    out, notes = drop_unpublishable_metadata(data, errors)
    assert out == {"license": "cc-by-4.0"}
    assert notes[-1] == "dropped empty datasets"


def test_a_card_the_hub_accepts_is_left_exactly_as_it_is():
    from espnet_model_zoo.hub_pipeline_tags import drop_unpublishable_metadata

    data = {"tags": ["espnet"], "language": "en", "license": "cc-by-4.0"}
    out, notes = drop_unpublishable_metadata(dict(data), [])
    assert out == data and notes == []


@pytest.mark.parametrize(
    "card", ["---\n- espnet\n---\n\nbody\n", "---\njust a string\n---\n\nbody\n"]
)
def test_front_matter_that_is_not_a_mapping_has_no_keys(card):
    from espnet_model_zoo.hub_pipeline_tags import _front_matter_keys

    assert _front_matter_keys(card) == {}
