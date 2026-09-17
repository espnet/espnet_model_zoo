from pathlib import Path

import pytest

from espnet_model_zoo.downloader import (
    ModelDownloader,
    cmd_download,
    cmd_query,
    download,
)


def test_download():
    download("http://example.com", "index.html")


def test_update_model_table(tmp_path):
    d = ModelDownloader(tmp_path)
    d.update_model_table()


def test_get_data_frame():
    d = ModelDownloader()
    d.get_data_frame()


def test_new_cachedir(tmp_path):
    ModelDownloader(tmp_path)


def test_download_and_unpack_names_with_condition():
    d = ModelDownloader()
    d.query("name", task="asr")


def test_get_model_names_and_urls():
    d = ModelDownloader()
    d.query(["name", "url"], task="asr")


def test_get_model_names_non_matching():
    d = ModelDownloader()
    assert d.query("name", task="dummy") == []


def test_download_and_unpack_with_url():
    d = ModelDownloader()
    d.download_and_unpack("https://zenodo.org/record/3951842/files/test.zip?download=1")


def test_download_and_unpack_with_name():
    d = ModelDownloader()
    d.download_and_unpack("test")


def test_download_and_unpack_no_inputting():
    d = ModelDownloader()
    with pytest.raises(TypeError):
        d.download_and_unpack()


def test_download_and_unpack_non_matching():
    d = ModelDownloader()
    with pytest.raises(RuntimeError):
        d.download_and_unpack(task="dummy")


def test_download_and_unpack_local_file():
    d = ModelDownloader()
    path = d.download("test")
    d.download_and_unpack(path)


def test_download_and_clean_cache():
    d = ModelDownloader()
    d.download_and_unpack("test")
    p = d.download("test")
    d.clean_cache("test")
    assert not Path(p).exists()


def test_cmd_download():
    cmd_download(["test"])


def test_query():
    cmd_query([])


def test_table_row_with_full_hub_url_is_downloaded_from_hub(tmp_path, monkeypatch):
    # espnet/Wangyou_Zhang_chime4_enh_train_enh_conv_tasnet_raw is one of the
    # table.csv rows whose url column holds the full repository URL rather
    # than the "huggingface.co" marker. Given the bare tag, download_and_unpack
    # used to miss the Hub check, call download(), get a snapshot directory
    # back and try to unpack it as an archive.
    tag = "espnet/Wangyou_Zhang_chime4_enh_train_enh_conv_tasnet_raw"
    d = ModelDownloader(tmp_path)
    assert d.get_url(tag) == f"https://huggingface.co/{tag}"
    seen = []

    def fake_huggingface_download(name=None, **kw):
        seen.append(name)
        return name

    monkeypatch.setattr(d, "huggingface_download", fake_huggingface_download)
    monkeypatch.setattr(
        d,
        "_unpack_cache_dir_for_huggingface",
        lambda cache_dir: {"cache_dir": cache_dir},
    )
    assert d.download_and_unpack(tag) == {"cache_dir": tag}
    assert seen == [tag]
    assert d.download(tag) == tag
    # A full URL passed as the name still works, and a marker row is untouched.
    assert d.download_and_unpack(f"https://huggingface.co/{tag}") == {"cache_dir": tag}
    marker_tag = "espnet/owsm_ctc_v4_1B"
    assert d.download_and_unpack(marker_tag) == {"cache_dir": marker_tag}
