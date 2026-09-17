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


def _fake_snapshot(root: Path):
    # The shape of a packed espnet model as huggingface_hub lays it out: a
    # meta.yaml naming the config and weights, a config whose paths are
    # relative to the repository root, and the files themselves.
    (root / "exp").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "exp" / "model.pth").write_bytes(b"w")
    (root / "exp" / "stats.npz").write_bytes(b"s")
    (root / "data" / "bpe.model").write_bytes(b"b")
    (root / "exp" / "config.yaml").write_text(
        "bpemodel: data/bpe.model\n"
        "normalize_conf:\n  stats_file: exp/stats.npz\n"
        "token_list:\n- <blank>\n- a\n"
        "frontend: default\n",
        encoding="utf-8",
    )
    (root / "meta.yaml").write_text(
        "files:\n  asr_model_file: exp/model.pth\n"
        "yaml_files:\n  asr_train_config: exp/config.yaml\n",
        encoding="utf-8",
    )


def test_hub_snapshot_stays_pristine_and_survives_a_move(tmp_path):
    import shutil

    import yaml

    snap = tmp_path / "snapshots" / "abc"
    _fake_snapshot(snap)
    pristine = (snap / "exp" / "config.yaml").read_text(encoding="utf-8")

    out = ModelDownloader._unpack_cache_dir_for_huggingface(str(snap))
    assert out["asr_model_file"] == str(snap / "exp" / "model.pth")
    assert out["asr_train_config"] == str(snap / "exp" / "config.resolved.yaml")
    resolved = yaml.safe_load(Path(out["asr_train_config"]).read_text())
    assert resolved["bpemodel"] == str(snap / "data" / "bpe.model")
    assert resolved["normalize_conf"]["stats_file"] == str(snap / "exp" / "stats.npz")
    assert (
        resolved["token_list"] == ["<blank>", "a"] and resolved["frontend"] == "default"
    )
    # the downloaded file is untouched
    assert (snap / "exp" / "config.yaml").read_text(encoding="utf-8") == pristine

    # Move the whole cache: the resolved config follows the new location.
    moved = tmp_path / "elsewhere" / "snapshots" / "abc"
    moved.parent.mkdir(parents=True)
    shutil.move(str(snap), str(moved))
    out = ModelDownloader._unpack_cache_dir_for_huggingface(str(moved))
    resolved = yaml.safe_load(Path(out["asr_train_config"]).read_text())
    assert resolved["bpemodel"] == str(moved / "data" / "bpe.model")
    assert resolved["normalize_conf"]["stats_file"] == str(moved / "exp" / "stats.npz")


def test_config_rewritten_in_place_by_an_older_version_is_healed(tmp_path):
    import yaml

    snap = tmp_path / "snapshots" / "abc"
    _fake_snapshot(snap)
    # What espnet_model_zoo <= 0.1.8 left behind: absolute paths into a cache
    # directory that no longer exists, written over the downloaded config.
    old = "/old/site-packages/espnet_model_zoo/models--x/snapshots/abc"
    (snap / "exp" / "config.yaml").write_text(
        f"bpemodel: {old}/data/bpe.model\n"
        f"normalize_conf:\n  stats_file: {old}/exp/stats.npz\n"
        "frontend: default\n",
        encoding="utf-8",
    )
    out = ModelDownloader._unpack_cache_dir_for_huggingface(str(snap))
    resolved = yaml.safe_load(Path(out["asr_train_config"]).read_text())
    assert resolved["bpemodel"] == str(snap / "data" / "bpe.model")
    assert resolved["normalize_conf"]["stats_file"] == str(snap / "exp" / "stats.npz")
    assert resolved["frontend"] == "default"


def test_paths_leading_out_of_the_snapshot_are_left_alone(tmp_path):
    import yaml

    snap = tmp_path / "snapshots" / "abc"
    _fake_snapshot(snap)
    sibling = tmp_path / "snapshots" / "other.npz"
    sibling.write_bytes(b"x")
    (snap / "exp" / "config.yaml").write_text(
        "bpemodel: data/bpe.model\n"
        "normalize_conf:\n  stats_file: ../other.npz\n"
        "frontend: default\n",
        encoding="utf-8",
    )
    out = ModelDownloader._unpack_cache_dir_for_huggingface(str(snap))
    resolved = yaml.safe_load(Path(out["asr_train_config"]).read_text())
    assert resolved["bpemodel"] == str(snap / "data" / "bpe.model")
    # exists, but outside the snapshot: not bound into the sidecar
    assert resolved["normalize_conf"]["stats_file"] == "../other.npz"


class _FakeResponse:
    def __init__(self, chunks, fail_after=None):
        self.headers = {"content-length": str(sum(len(c) for c in chunks))}
        self._chunks, self._fail_after = chunks, fail_after

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i, c in enumerate(self._chunks):
            if self._fail_after is not None and i == self._fail_after:
                raise ConnectionError("cut")
            yield c


class _FakeSession:
    response = None

    def mount(self, *a, **k):
        pass

    def get(self, url, stream, timeout):
        return self.response


def test_download_writes_beside_the_target_and_renames(tmp_path, monkeypatch):
    import requests

    _FakeSession.response = _FakeResponse([b"abc", b"def"])
    monkeypatch.setattr(requests, "Session", _FakeSession)
    out = tmp_path / "models" / "m.zip"
    download("http://x/m.zip", out, quiet=True)
    assert out.read_bytes() == b"abcdef"
    # no part file left next to it
    assert [p.name for p in out.parent.iterdir()] == ["m.zip"]


def test_interrupted_download_leaves_nothing_behind(tmp_path, monkeypatch):
    import requests

    _FakeSession.response = _FakeResponse([b"abc", b"def"], fail_after=1)
    monkeypatch.setattr(requests, "Session", _FakeSession)
    out = tmp_path / "m.zip"
    with pytest.raises(ConnectionError):
        download("http://x/m.zip", out, quiet=True)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []  # the .part file is gone too


def test_file_name_is_a_bare_name_whatever_the_server_says(monkeypatch):
    import requests

    class Head:
        headers = {"Content-Disposition": 'attachment; filename="../../evil.zip"'}

    monkeypatch.setattr(requests, "head", lambda url, **kw: Head())
    assert ModelDownloader._get_file_name("http://x/dl") == "evil.zip"
    zenodo = "https://zenodo.org/record/1/files/asr_train.zip?download=1"
    assert ModelDownloader._get_file_name(zenodo) == "asr_train.zip"
    Head.headers = {}
    assert ModelDownloader._get_file_name("http://x/path/model.tgz?a=1") == "model.tgz"
    Head.headers = {"Content-Disposition": "attachment; filename=.."}
    with pytest.raises(ValueError):
        ModelDownloader._get_file_name("http://x/")
    # download() keeps its own "url" note and unpack() its meta.yaml in the
    # same directory; an archive under either name would be clobbered
    Head.headers = {"Content-Disposition": "attachment; filename=url"}
    assert ModelDownloader._get_file_name("http://x/dl") == "download_url"
    Head.headers = {"Content-Disposition": "attachment; filename=meta.yaml"}
    assert ModelDownloader._get_file_name("http://x/dl") == "download_meta.yaml"
