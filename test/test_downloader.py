from pathlib import Path
import pytest

from espnet_model_zoo.downloader import cmd_download
from espnet_model_zoo.downloader import cmd_query
from espnet_model_zoo.downloader import download
from espnet_model_zoo.downloader import ModelDownloader


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
