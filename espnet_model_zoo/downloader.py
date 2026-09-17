import argparse
import hashlib
import re
import shutil
import tempfile
import warnings
from distutils.util import strtobool
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import pandas as pd
import requests
import yaml
from espnet2.main_funcs.pack_funcs import (
    get_dict_from_cache,
    unpack,
)
from filelock import FileLock
from huggingface_hub import snapshot_download
from tqdm import tqdm

MODELS_URL = (
    "https://raw.githubusercontent.com/espnet/espnet_model_zoo/master/"
    "espnet_model_zoo/table.csv"
)


URL_REGEX = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)"
    r"+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def is_url(url: str) -> bool:
    return re.match(URL_REGEX, url) is not None


def str_to_hash(string: Union[str, Path]) -> str:
    return hashlib.md5(str(string).encode("utf-8")).hexdigest()


def download(
    url, output_path, retry: int = 3, chunk_size: int = 8192, quiet: bool = False
):
    # Set retry
    session = requests.Session()
    session.mount("http://", requests.adapters.HTTPAdapter(max_retries=retry))
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry))

    # Timeout
    response = session.get(url=url, stream=True, timeout=(10.0, 30.0))

    # Raise error when connection error
    response.raise_for_status()

    # Only the progress bar wants the size; a chunked response has none.
    file_size = response.headers.get("content-length")
    file_size = int(file_size) if file_size is not None else None

    # Write in temporary file
    with tempfile.TemporaryDirectory() as d:
        with (Path(d) / "tmp").open("wb") as f:
            if quiet:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            else:
                with tqdm(
                    desc=url,
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(Path(d) / "tmp", output_path)


_HF_MARKERS = ("https://huggingface.co/", "https://huggingface.co", "huggingface.co")


def _resolve_huggingface(name, url):
    """Return (name, url) with any Hugging Face reference in canonical form.

    A model reaches the downloader in one of three shapes: a bare Hub id
    ("espnet/x") whose table.csv row says just "huggingface.co"; a full
    repository URL given as the name; or a bare id whose table.csv row holds
    the full URL, which is how ten espnet-organisation rows were entered. The
    first two were handled and the third was not: the url failed the marker
    test here, download() then recognised it and returned a snapshot
    directory, and download_and_unpack() tried to unpack that directory as an
    archive. All three now come out as (repo_id, "https://huggingface.co/").
    """
    for candidate in (name, url):
        if isinstance(candidate, str) and candidate.startswith(
            "https://huggingface.co/"
        ):
            repo_id = candidate[len("https://huggingface.co/") :]
            return repo_id, "https://huggingface.co/"
    return name, url


def _resolve_paths(value, root: Path):
    """Make the paths in a packed config absolute under ``root``.

    A string that names a file or directory relative to ``root`` becomes that
    absolute path. An absolute string that no longer exists - a config that an
    older version rewrote for a directory since moved - is looked up by its
    tail: the longest trailing part of it that exists under ``root`` wins, so
    ``/old/snapshots/abc/exp/stats.npz`` becomes ``root/exp/stats.npz``.
    Anything else is returned unchanged.
    """
    if isinstance(value, dict):
        return {k: _resolve_paths(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolve_paths(v, root) for v in value]
    if not isinstance(value, str) or not value:
        return value
    candidate = Path(value)
    if not candidate.is_absolute():
        return str(root / value) if (root / value).exists() else value
    if candidate.exists():
        return value
    parts = candidate.parts
    for i in range(1, len(parts)):
        healed = root.joinpath(*parts[i:])
        if healed.exists():
            return str(healed)
    return value


class ModelDownloader:
    """Download model from zenodo and unpack."""

    def __init__(self, cachedir: Union[Path, str] = None):
        # A None cachedir used to mean this package's own directory, so a
        # pip-installed espnet_model_zoo grew by the size of every model it
        # fetched (4 GB for one OWSM checkpoint) inside site-packages, and
        # `pip uninstall` took the models with it. The home cache that was the
        # fallback for a read-only install is now the default for every
        # install; pass cachedir to keep the models somewhere else.
        self._explicit_cachedir = cachedir is not None
        if cachedir is None:
            cachedir = Path.home() / ".cache" / "espnet_model_zoo"
        else:
            cachedir = Path(cachedir).expanduser().absolute()
        cachedir.mkdir(parents=True, exist_ok=True)

        csv = Path(__file__).parent / "table.csv"
        if not csv.exists():
            download(MODELS_URL, csv)

        self.cachedir = cachedir
        self.csv = csv
        self.data_frame = pd.read_csv(csv, dtype=str)

    def get_data_frame(self):
        return self.data_frame

    def update_model_table(self):
        lock_file = str(self.csv) + ".lock"
        Path(lock_file).parent.mkdir(parents=True, exist_ok=True)
        with FileLock(lock_file):
            download(MODELS_URL, self.csv)

    def clean_cache(self, name: str = None, version: int = -1, **kwargs: str):
        url = self.get_url(name=name, version=version, **kwargs)
        outdir = self.cachedir / str_to_hash(url)
        shutil.rmtree(outdir)

    def query(
        self, key: Union[Sequence[str]] = "name", **kwargs
    ) -> List[Union[str, Tuple[str]]]:
        conditions = None
        for k, v in kwargs.items():
            if k not in self.data_frame:
                warnings.warn(
                    f"Invalid key: {k}: Available keys:\n"
                    f"{list(self.data_frame.keys())}"
                )
                continue
            condition = self.data_frame[k] == v
            if conditions is None:
                conditions = condition
            else:
                conditions &= condition

        if conditions is not None:
            df = self.data_frame[conditions]
        else:
            df = self.data_frame

        if len(df) == 0:
            return []
        else:
            if isinstance(key, (tuple, list)):
                return list(zip(*[df[k] for k in key]))
            else:
                return list(df[key])

    def get_url(self, name: str = None, version: int = -1, **kwargs: str) -> str:
        if name is None and len(kwargs) == 0:
            raise TypeError("No arguments are given")

        if name is not None and is_url(name):
            # Specify the downloading link directly. "kwargs" are ignored in this case.
            url = name

        else:
            if name is not None:
                kwargs["name"] = name

            conditions = None
            for key, value in kwargs.items():
                condition = self.data_frame[key] == value
                if conditions is None:
                    conditions = condition
                else:
                    conditions &= condition

            if len(self.data_frame[conditions]) == 0:
                # If Specifying local file path
                if name is not None and Path(name).exists() and len(kwargs) == 1:
                    url = str(Path(name).absolute())

                else:
                    return "huggingface.co"
            else:
                urls = self.data_frame[conditions]["url"]
                if version < 0:
                    version = len(urls) + version
                url = list(urls)[version]
        return url

    @staticmethod
    def _get_file_name(url):
        ma = re.match(r"https://.*/([^/]*)\?download=[0-9]*$", url)
        if ma is not None:
            # URL e.g.
            # https://sandbox.zenodo.org/record/646767/files/asr_train_raw_bpe_valid.acc.best.zip?download=1
            a = ma.groups()[0]
            return a
        else:
            # If not Zenodo
            r = requests.head(url)
            if "Content-Disposition" in r.headers:
                # e.g. attachment; filename=asr_train_raw_bpe_valid.acc.best.zip
                for v in r.headers["Content-Disposition"].split(";"):
                    if "filename=" in v:
                        return v.split("filename=")[1].strip()

            # if not specified or some error happens
            return Path(url).name

    def unpack_local_file(self, name: str = None) -> Dict[str, Union[str, List[str]]]:
        if not Path(name).exists():
            raise FileNotFoundError(f"No such file or directory: {name}")

        warnings.warn(
            "Expanding a local model to the cachedir. "
            "If you'll move the file to another path, "
            "it's treated as a different model."
        )
        name = Path(name).absolute()

        outdir = self.cachedir / str_to_hash(name)
        filename = outdir / name.name
        outdir.mkdir(parents=True, exist_ok=True)

        if not filename.exists():
            if filename.is_symlink():
                filename.unlink()
            filename.symlink_to(name)

        # Skip unpacking if the cache exists
        meta_yaml = outdir / "meta.yaml"
        outdir.mkdir(parents=True, exist_ok=True)
        lock_file = str(meta_yaml) + ".lock"
        with FileLock(lock_file):
            if meta_yaml.exists():
                info = get_dict_from_cache(meta_yaml)
                if info is not None:
                    return info

            # Extract files from archived file
            return unpack(filename, outdir)

    def huggingface_download(
        self, name: str = None, version: int = -1, quiet: bool = False, **kwargs: str
    ) -> str:
        # Get huggingface_id from table.csv
        if name is None:
            names = self.query(key="name", **kwargs)
            if len(names) == 0:
                message = "Not found models:"
                for key, value in kwargs.items():
                    message += f" {key}={value}"
                raise RuntimeError(message)
            if version < 0:
                version = len(names) + version
            name = list(names)[version]

        if "@" in name:
            huggingface_id, revision = name.split("@", 1)
        else:
            huggingface_id = name
            revision = None

        # Without an explicit cachedir, let huggingface_hub use its own cache
        # (HF_HOME / HF_HUB_CACHE), so a model already fetched by transformers
        # or `hf download` is not downloaded a second time here.
        return snapshot_download(
            huggingface_id,
            revision=revision,
            library_name="espnet",
            cache_dir=self.cachedir if self._explicit_cachedir else None,
        )

    @staticmethod
    def _unpack_cache_dir_for_huggingface(cache_dir: str):
        """Return constructor kwargs for a Hub snapshot, with paths resolved.

        A packed model's config refers to its files by paths relative to the
        repository root (``bpemodel: data/bpe.model``). Until now this method
        rewrote those in place into absolute paths under ``cache_dir`` - once,
        guarded by a ``.done`` marker - which mutated the snapshot (a symlink
        into huggingface_hub's blob store, so the blob itself changed) and
        froze the location: moving or copying the cache broke every model in
        it, since the config then pointed at the old directory.

        The snapshot is now left as downloaded. Each config is written once
        more as ``<name>.resolved.yaml`` next to it with the paths made
        absolute, regenerated whenever ``cache_dir`` is not the directory the
        resolved files were written for, and a config that an older version
        rewrote in place is healed by locating each missing absolute path by
        its tail under the current ``cache_dir``.
        """
        cache_dir = Path(cache_dir)
        meta_yaml = cache_dir / "meta.yaml"
        lock_file = cache_dir / ".lock"
        root_file = cache_dir / ".resolved_root"

        with meta_yaml.open("r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
            assert isinstance(d, dict), type(d)
            yaml_files = d["yaml_files"]
            files = d["files"]
            assert isinstance(yaml_files, dict), type(yaml_files)
            assert isinstance(files, dict), type(files)

        retval = {}
        with FileLock(lock_file):
            stale = not root_file.exists() or root_file.read_text(
                encoding="utf-8"
            ).strip() != str(cache_dir)
            for key, value in yaml_files.items():
                src = cache_dir / value
                dst = src.with_name(src.stem + ".resolved" + src.suffix)
                if stale or not dst.exists():
                    with src.open("r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                        assert isinstance(config, dict), type(config)
                    with dst.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(_resolve_paths(config, cache_dir), f)
                retval[key] = str(dst)
            root_file.write_text(str(cache_dir), encoding="utf-8")

        for key, value in files.items():
            retval[key] = str(cache_dir / value)
        return retval

    def download(
        self, name: str = None, version: int = -1, quiet: bool = False, **kwargs: str
    ) -> str:
        url = self.get_url(name=name, version=version, **kwargs)

        name, url = _resolve_huggingface(name, url)
        if url in _HF_MARKERS:
            # TODO(kamo): Support quiet
            cache_dir = self.huggingface_download(name=name, version=version, **kwargs)
            self._unpack_cache_dir_for_huggingface(cache_dir)
            return cache_dir

        if not is_url(url) and Path(url).exists():
            return url

        outdir = self.cachedir / str_to_hash(url)
        filename = self._get_file_name(url)
        # Download the model file if not existing
        outdir.mkdir(parents=True, exist_ok=True)
        lock_file = str(outdir / filename) + ".lock"
        with FileLock(lock_file):
            if not (outdir / filename).exists():
                download(url, outdir / filename, quiet=quiet)

                # Write the url for debugging
                with (outdir / "url").open("w", encoding="utf-8") as f:
                    f.write(url)

                r = requests.head(url)
                if "Content-MD5" in r.headers:
                    checksum = r.headers["Content-MD5"]

                    # MD5 checksum
                    sig = hashlib.md5()
                    chunk_size = 8192
                    with open(outdir / filename, "rb") as f:
                        while True:
                            chunk = f.read(chunk_size)
                            if len(chunk) == 0:
                                break
                            sig.update(chunk)

                    if sig.hexdigest() != checksum:
                        Path(outdir / filename).unlink()
                        raise RuntimeError(f"Failed to download file: {url}")
                else:
                    warnings.warn("Not validating checksum")
        return str(outdir / filename)

    def download_and_unpack(
        self, name: str = None, version: int = -1, quiet: bool = False, **kwargs: str
    ) -> Dict[str, Union[str, List[str]]]:
        url = self.get_url(name=name, version=version, **kwargs)
        if not is_url(url) and Path(url).exists():
            return self.unpack_local_file(url)

        name, url = _resolve_huggingface(name, url)
        if url in _HF_MARKERS:
            # download_and_unpack and download are same if huggingface case
            # TODO(kamo): Support quiet
            cache_dir = self.huggingface_download(name=name, version=version, **kwargs)
            return self._unpack_cache_dir_for_huggingface(cache_dir)

        # Unpack to <cachedir>/<hash> in order to give an unique name
        outdir = self.cachedir / str_to_hash(url)

        # Skip downloading and unpacking if the cache exists
        meta_yaml = outdir / "meta.yaml"
        outdir.mkdir(parents=True, exist_ok=True)
        lock_file = str(meta_yaml) + ".lock"
        with FileLock(lock_file):
            if meta_yaml.exists():
                info = get_dict_from_cache(meta_yaml)
                if info is not None:
                    return info

            # Download the file to an unique path
            filename = self.download(url, quiet=quiet)

            # Extract files from archived file
            return unpack(filename, outdir)


def str2bool(v) -> bool:
    return bool(strtobool(v))


def cmd_download(cmd=None):
    # espnet_model_zoo_download

    parser = argparse.ArgumentParser("Download file from Zenodo")
    parser.add_argument(
        "name",
        help="URL or model name in the form of <username>/<model name>. "
        "e.g. kamo-naoyuki/mini_an4_asr_train_raw_bpe_valid.acc.best",
    )
    parser.add_argument(
        "--cachedir",
        help="Specify cache dir. By default, download to module root.",
    )
    parser.add_argument(
        "--unpack",
        type=str2bool,
        default=False,
        help="Unpack the archived file after downloading.",
    )
    args = parser.parse_args(cmd)

    d = ModelDownloader(args.cachedir)
    if args.unpack:
        print(d.download_and_unpack(args.name))
    else:
        print(d.download(args.name))


def cmd_query(cmd=None):
    # espnet_model_zoo_query

    parser = argparse.ArgumentParser("Download file from Zenodo")

    parser.add_argument(
        "condition",
        nargs="*",
        default=[],
        help="Given desired condition in form of <key>=<value>. "
        "e.g. fs=16000. "
        "If no condition is given, you can view all available models",
    )
    parser.add_argument(
        "--key",
        default="name",
        help="The key name you want",
    )
    parser.add_argument(
        "--cachedir",
        help="Specify cache dir. By default, download to module root.",
    )
    args = parser.parse_args(cmd)

    conditions = dict(s.split("=") for s in args.condition)
    d = ModelDownloader(args.cachedir)
    for v in d.query(args.key, **conditions):
        print(v)
