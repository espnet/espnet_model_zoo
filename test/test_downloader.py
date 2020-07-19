import pytest

from espnet_model_zoo.downloader import download
from espnet_model_zoo.downloader import ModelDownloader


def test_download():
    download("http://example.com", "index.html")


def test_update_model_table():
    d = ModelDownloader()
    d.update_model_table()


def test_get_data_frame():
    d = ModelDownloader()
    d.get_data_frame()


def test_new_cachedir(tmp_path):
    ModelDownloader(tmp_path)


def test_get_model_names_with_condition():
    d = ModelDownloader()
    d.query("name", task="asr")


def test_get_model_names_and_urls():
    d = ModelDownloader()
    d.query(["name", "url"], task="asr")


def test_get_model_names_non_matching():
    d = ModelDownloader()
    assert d.query("name", task="dummy") == []


def test_get_model_with_url():
    d = ModelDownloader()
    d.get_model("https://zenodo.org/record/3951085/files/test.zip?download=1")


def test_get_model_with_name():
    d = ModelDownloader()
    d.get_model("test")


def test_get_model_no_inputting():
    d = ModelDownloader()
    with pytest.raises(TypeError):
        d.get_model()


def test_get_model_non_matching():
    d = ModelDownloader()
    with pytest.raises(RuntimeError):
        d.get_model(task="dummy")
