import json

import pytest

from espnet_model_zoo import zenodo_upload
from espnet_model_zoo.zenodo_upload import Zenodo, upload


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


def test_files_are_checked_before_anything_is_created(tmp_path, monkeypatch):
    created = []
    monkeypatch.setattr(Zenodo, "create_deposition", lambda self: created.append(1))
    with pytest.raises(FileNotFoundError):
        upload("tok", "t", "me", files=[tmp_path])  # a directory, not a file
    with pytest.raises(FileNotFoundError):
        upload("tok", "t", "me", files=[tmp_path / "missing.zip"])
    assert created == []  # no draft left behind on Zenodo


def test_community_goes_inside_metadata(tmp_path, monkeypatch):
    sent = {}
    r = _Resp(201, {"id": 7, "links": {"html": "h", "bucket": "b"}})
    monkeypatch.setattr(Zenodo, "create_deposition", lambda self: r)
    monkeypatch.setattr(
        Zenodo, "update_metadata", lambda self, r, data: sent.update(data)
    )
    monkeypatch.setattr(Zenodo, "upload_file", lambda self, r, f: None)
    f = tmp_path / "m.zip"
    f.write_bytes(b"x")
    upload("tok", "t", "me", files=[f], community_identifer="espnet")
    assert sent["metadata"]["communities"] == [{"identifier": "espnet"}]
    assert "communities" not in sent


def test_every_zenodo_request_carries_a_timeout(tmp_path, monkeypatch):
    seen = []

    calls = []

    def fake(method):
        def call(url, **kw):
            seen.append((method, kw.get("timeout")))
            calls.append((method, kw))
            body = {"id": 7, "links": {"bucket": "http://b", "latest_html": "x"}}
            return _Resp({"post": 201, "get": 200, "put": 200}[method], body)

        return call

    for m in ("post", "get", "put"):
        monkeypatch.setattr(zenodo_upload.requests, m, fake(m))
    z = Zenodo("tok")
    r = z.create_deposition()
    z.get_deposition(r)
    z.update_metadata(r, {"metadata": {}})
    f = tmp_path / "m.zip"
    f.write_bytes(b"x")
    z.upload_file(r, f)
    z.upload_file(7, f)
    # the by-id branch must authenticate: every GET carried the access token
    assert all("params" in kw for m, kw in calls if m == "get"), calls
    # publish answers 202
    monkeypatch.setattr(
        zenodo_upload.requests,
        "post",
        lambda url, **kw: (
            seen.append(("post", kw.get("timeout"))),
            _Resp(202, {"links": {"latest_html": "x"}}),
        )[1],
    )
    z.publish(r)
    assert seen and all(t is not None for _, t in seen), seen
    assert ("put", zenodo_upload.UPLOAD_TIMEOUT) in seen
    json.dumps(seen)  # every timeout is a plain tuple
