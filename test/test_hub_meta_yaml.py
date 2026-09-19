import pytest

from espnet_model_zoo.hub_meta_yaml import (
    checkpoint_rank,
    choose_pair,
    meta_yaml,
    missing_inputs,
    symlink_target,
    task_of,
)


def test_the_config_says_the_task_even_when_the_name_does_not():
    assert task_of({"tts": "vits", "tts_conf": {}}, "exp/whatever") == "tts"
    assert task_of({"separator": "tfgridnet"}, "") == "enh"
    assert task_of({"attractor": "rnn"}, "") == "diar"
    assert task_of({"corelm": "valle"}, "") == "speechlm"
    assert task_of({"pooling": "chn_attn_stat", "projector": "rawnet3"}, "") == "spk"


def test_the_experiment_directory_decides_what_the_config_does_not():
    assert task_of({}, "exp/s2t_train_1b_ds_90k_raw_bpe50000") == "s2t"
    assert task_of({}, "exp/vad_train_vad_rnn_raw") == "vad"
    assert task_of({"ctc_conf": {}, "decoder": "transformer"}, "exp/run") == "asr"
    assert task_of({}, "exp/unknown") == ""


def test_a_stats_config_is_not_the_training_config():
    # the repository holds the statistics run's config at the shallower path;
    # the one beside the checkpoint is the model's
    files = [
        "exp/asr_stats_raw_zh_char_sp/logdir/stats.1/config.yaml",
        "exp/asr_stats_raw_zh_char_sp/logdir/stats.1/train/feats_stats.npz",
        "exp/asr_train_asr_conformer_raw_zh_char_sp/config.yaml",
        "exp/asr_train_asr_conformer_raw_zh_char_sp/valid.acc.ave_10best.pth",
    ]
    config, checkpoint, split = choose_pair(files)
    assert config == "exp/asr_train_asr_conformer_raw_zh_char_sp/config.yaml"
    assert checkpoint.endswith("valid.acc.ave_10best.pth")
    assert split is False


def test_a_checkpoint_in_the_next_directory_is_paired_but_flagged():
    files = [
        "exp_11k/s2t_train_1b_ds_11k_2_raw_bpe50000/config.yaml",
        "exp_11k/s2t_train_1b_ds_raw_bpe50000/valid.total_count.ave_5best.pth",
    ]
    config, checkpoint, split = choose_pair(files)
    assert config.endswith("config.yaml") and checkpoint.endswith(".pth")
    assert split is True


def test_nothing_to_pair():
    assert choose_pair(["README.md"]) == ("", "", False)
    assert choose_pair(["config.yaml"]) == ("", "", False)


def test_the_averaged_checkpoint_wins_and_the_resume_state_loses():
    order = sorted(
        [
            "exp/a/checkpoint.pth",
            "exp/a/30epoch.pth",
            "exp/a/40epoch.pth",
            "exp/a/valid.acc.ave_10best.pth",
        ],
        key=checkpoint_rank,
    )
    assert order[0] == "exp/a/valid.acc.ave_10best.pth"
    assert order[1] == "exp/a/40epoch.pth"  # the later epoch before the earlier
    assert order[-1] == "exp/a/checkpoint.pth"


def test_a_symlink_uploaded_as_text_is_followed():
    target = symlink_target("exp/a/valid.acc.ave.pth", "valid.acc.ave_10best.pth\n")
    assert target == "exp/a/valid.acc.ave_10best.pth"


@pytest.mark.parametrize("text", ["PK\x03\x04binary", "one.pth\ntwo.pth", "notes.txt"])
def test_only_a_single_checkpoint_name_counts_as_a_symlink(text):
    assert symlink_target("exp/a/valid.acc.ave.pth", text) is None


def test_the_vocabulary_is_not_a_missing_file():
    config = {"token_list": ["<blank>", "<sos/eos>", "a/b"], "bpemodel": None}
    assert missing_inputs(config, ["exp/a/config.yaml"]) == []


def test_a_tokenizer_on_the_hub_is_not_a_missing_file():
    config = {"subword_model": "HuggingFaceTB/SmolLM-1.7B"}
    assert missing_inputs(config, []) == []


def test_a_bpemodel_the_repository_does_not_hold_is_reported():
    config = {
        "bpemodel": "data/en_token_list/bpe_unigram500/bpe.model",
        "normalize_conf": {"stats_file": "exp/stats/train/feats_stats.npz"},
    }
    missing = missing_inputs(config, ["exp/stats/train/feats_stats.npz"])
    assert missing == ["bpemodel=data/en_token_list/bpe_unigram500/bpe.model"]


def test_a_path_the_config_writes_with_a_leading_dot_still_counts_as_present():
    config = {"bpemodel": "./data/bpe.model"}
    assert missing_inputs(config, ["data/bpe.model"]) == []


def test_the_keys_are_the_ones_the_inference_class_takes():
    assert "asr_model_file: exp/a.pth" in meta_yaml("asr", "exp/c.yaml", "exp/a.pth")
    assert "asr_train_config: exp/c.yaml" in meta_yaml("asr", "exp/c.yaml", "exp/a.pth")
    # tts, enh, diar, spk and the rest pack under the generic pair
    generic = meta_yaml("enh", "config.yaml", "valid.loss.best.pth")
    assert "model_file: valid.loss.best.pth" in generic
    assert "train_config: config.yaml" in generic
