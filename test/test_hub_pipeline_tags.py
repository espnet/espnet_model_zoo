import pytest

from espnet_model_zoo.hub_pipeline_tags import infer

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
