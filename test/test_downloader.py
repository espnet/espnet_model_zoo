from pathlib import Path

import pytest
import yaml

from espnet_model_zoo.downloader import (
    ModelDownloader,
    _resolve_paths,
    _unresolve_paths,
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


def test_unpack_without_meta_yaml_names_the_files(tmp_path):
    (tmp_path / "config.yaml").write_text("encoder: stft\n")
    (tmp_path / "valid.loss.best.pth").write_bytes(b"")
    with pytest.raises(RuntimeError, match=r"no meta\.yaml") as e:
        ModelDownloader._unpack_cache_dir_for_huggingface(str(tmp_path))
    assert "config.yaml, valid.loss.best.pth" in str(e.value)
    assert "train_config=" in str(e.value)


def _snapshot(tmp_path):
    """A snapshot laid out the way pack_model writes one."""
    (tmp_path / "exp" / "asr_stats" / "train").mkdir(parents=True)
    (tmp_path / "exp" / "asr_stats" / "train" / "feats_stats.npz").write_bytes(b"")
    (tmp_path / "data" / "token_list").mkdir(parents=True)
    (tmp_path / "data" / "token_list" / "bpe.model").write_bytes(b"")
    return tmp_path


def test_resolve_paths_keeps_the_real_references(tmp_path):
    root = _snapshot(tmp_path)
    config = {
        "normalize_conf": {"stats_file": "exp/asr_stats/train/feats_stats.npz"},
        "bpemodel": "data/token_list/bpe.model",
    }
    out = _resolve_paths(config, root)
    assert out["normalize_conf"]["stats_file"] == str(
        root / "exp/asr_stats/train/feats_stats.npz"
    )
    assert out["bpemodel"] == str(root / "data/token_list/bpe.model")


def test_resolve_paths_leaves_vocabulary_entries_alone(tmp_path):
    root = _snapshot(tmp_path)
    # "." and "exp" name real things inside the snapshot, and both are pieces
    # in OWSM's vocabulary; rewriting them put the cache directory into every
    # transcript
    config = {
        "token_list": [".", "exp", "data", "▁the", "s"],
        "brctc_risk_strategy": "exp",
        "recipe_dir": ".",
    }
    assert _resolve_paths(config, root) == config


def test_resolve_paths_heals_a_stale_absolute_path(tmp_path):
    root = _snapshot(tmp_path)
    old = "/somewhere/else/snapshots/abc/exp/asr_stats/train/feats_stats.npz"
    assert _resolve_paths({"stats_file": old}, root) == {
        "stats_file": str(root / "exp/asr_stats/train/feats_stats.npz")
    }


def test_resolve_paths_does_not_heal_down_to_a_bare_directory(tmp_path):
    root = _snapshot(tmp_path)
    # the tail "exp" alone would resolve to <root>/exp, which is how a token
    # became a directory name
    stale = "/gone/exp"
    assert _resolve_paths({"x": stale}, root) == {"x": stale}


def test_unpack_rewrites_a_sidecar_left_by_an_older_resolver(tmp_path):
    root = _snapshot(tmp_path)
    (root / "meta.yaml").write_text(
        "files: {}\nyaml_files:\n  train_config: exp/config.yaml\n", encoding="utf-8"
    )
    (root / "exp" / "config.yaml").write_text(
        "token_list:\n- '.'\n- exp\n", encoding="utf-8"
    )
    corrupted = root / "exp" / "config.resolved.yaml"
    corrupted.write_text(f"token_list:\n- {root}\n- {root}/exp\n", encoding="utf-8")
    (root / ".resolved_root").write_text(str(root), encoding="utf-8")

    out = ModelDownloader._unpack_cache_dir_for_huggingface(str(root))

    with open(out["train_config"], encoding="utf-8") as f:
        assert yaml.safe_load(f) == {"token_list": [".", "exp"]}


def test_unresolve_paths_restores_what_an_older_version_prefixed(tmp_path):
    root = _snapshot(tmp_path)
    config = {
        "token_list": [".", f"{root}/exp", "▁the"],
        "brctc_risk_strategy": f"{root}/exp",
        "stats_file": f"{root}/exp/asr_stats/train/feats_stats.npz",
        "output_dir": str(root),
        "untouched": "/somewhere/else/x",
    }
    assert _unresolve_paths(config, root) == {
        "token_list": [".", "exp", "▁the"],
        "brctc_risk_strategy": "exp",
        "stats_file": "exp/asr_stats/train/feats_stats.npz",
        "output_dir": ".",
        "untouched": "/somewhere/else/x",
    }


def test_unpack_repairs_a_config_an_older_version_rewrote(tmp_path):
    root = _snapshot(tmp_path)
    (root / "meta.yaml").write_text(
        "files: {}\nyaml_files:\n  train_config: exp/config.yaml\n", encoding="utf-8"
    )
    # the shape the in-place rewriting left behind: a vocabulary entry and an
    # option turned into the snapshot's exp directory, beside a real reference
    (root / "exp" / "config.yaml").write_text(
        f"token_list:\n- '.'\n- {root}/exp\n"
        f"brctc_risk_strategy: {root}/exp\n"
        f"bpemodel: {root}/data/token_list/bpe.model\n",
        encoding="utf-8",
    )

    out = ModelDownloader._unpack_cache_dir_for_huggingface(str(root))

    with open(out["train_config"], encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["token_list"] == [".", "exp"]
    assert config["brctc_risk_strategy"] == "exp"
    assert config["bpemodel"] == str(root / "data/token_list/bpe.model")


def test_unresolve_paths_strips_a_prefix_from_a_moved_snapshot(tmp_path):
    root = _snapshot(tmp_path)
    # damage done while the cache lived somewhere else
    old_root = "/somewhere/else/snapshots/abc"
    config = {
        "token_list": [".", f"{old_root}/exp"],
        "brctc_risk_strategy": f"{old_root}/exp",
        "stats_file": f"{old_root}/exp/asr_stats/train/feats_stats.npz",
    }
    assert _unresolve_paths(config, root) == {
        "token_list": [".", "exp"],
        "brctc_risk_strategy": "exp",
        "stats_file": "exp/asr_stats/train/feats_stats.npz",
    }


def test_unresolve_paths_keeps_an_absolute_path_with_no_counterpart(tmp_path):
    root = _snapshot(tmp_path)
    outside = "/opt/models/whatever/model.pth"
    assert _unresolve_paths({"x": outside}, root) == {"x": outside}


def test_unpack_repairs_a_config_rewritten_before_the_cache_moved(tmp_path):
    root = _snapshot(tmp_path)
    (root / "meta.yaml").write_text(
        "files: {}\nyaml_files:\n  train_config: exp/config.yaml\n", encoding="utf-8"
    )
    (root / "exp" / "config.yaml").write_text(
        "token_list:\n- '.'\n- /gone/snapshots/abc/exp\n"
        "bpemodel: /gone/snapshots/abc/data/token_list/bpe.model\n",
        encoding="utf-8",
    )

    out = ModelDownloader._unpack_cache_dir_for_huggingface(str(root))

    with open(out["train_config"], encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["token_list"] == [".", "exp"]
    assert config["bpemodel"] == str(root / "data/token_list/bpe.model")
