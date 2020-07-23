import argparse
from distutils.util import strtobool
import hashlib
from pathlib import Path
import re
import shutil
import tempfile
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union
import warnings

import pandas as pd
import requests
from tqdm import tqdm

from espnet2.main_funcs.pack_funcs import get_dict_from_cache
from espnet2.main_funcs.pack_funcs import unpack


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


def str_to_hash(string: str) -> str:
    return hashlib.md5(string.encode("utf-8")).hexdigest()


def download(url, output_path, retry: int = 3, chunk_size: int = 8192):
    # Set retry
    session = requests.Session()
    session.mount("http://", requests.adapters.HTTPAdapter(max_retries=retry))
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry))

    # Timeout
    response = session.get(url=url, stream=True, timeout=(10.0, 30.0))
    file_size = int(response.headers["content-length"])

    # Raise error when connection error
    response.raise_for_status()

    # Write in temporary file
    with tempfile.TemporaryDirectory() as d:
        with (Path(d) / "tmp").open("wb") as f:
            with tqdm(
                desc=url, total=file_size, unit="B", unit_scale=True, unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(Path(d) / "tmp", output_path)


class ModelDownloader:
    """Download model from zenodo and unpack."""

    def __init__(self, cachedir: Union[Path, str] = None):
        if cachedir is None:
            cachedir = Path(__file__).parent
        else:
            cachedir = Path(cachedir).expanduser().absolute()
        cachedir.mkdir(parents=True, exist_ok=True)

        csv = Path(__file__).parent / "table.csv"
        if not csv.exists():
            download(MODELS_URL, csv)

        self.cachedir = cachedir
        self.csv = csv
        self.data_frame = pd.read_csv(csv)

    def get_data_frame(self):
        return self.data_frame

    def update_model_table(self):
        download(MODELS_URL, self.csv)

    def query(
        self, key: Union[Sequence[str]] = "name", **kwargs
    ) -> List[Union[str, Tuple[str]]]:
        conditions = None
        for k, v in kwargs.items():
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

        if name is not None:
            kwargs["name"] = name

        conditions = None
        for key, value in kwargs.items():
            condition = self.data_frame[key] == value
            if conditions is None:
                conditions = condition
            else:
                conditions &= condition

        # If no models satisfy the conditions, raise an error
        if len(self.data_frame[conditions]) == 0:
            message = f"Not found models: name={name}"
            for key, value in kwargs.items():
                message += f", {key}={value}"
            raise RuntimeError(message)

        urls = self.data_frame[conditions]["url"]
        if version < 0:
            version = len(urls) + version
        url = list(urls)[version]
        return url

    def _get_file_name(self, url):
        ma = re.match(r"https://.*/([^/]*)\?download=[0-9]*$", url)
        if ma is not None:
            # URL e.g.
            # https://sandbox.zenodo.org/record/646767/files/asr_train_raw_bpe_valid.acc.best.zip?download=1
            return ma.groups()[0]
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

    def download(self, name: str = None, version: int = -1, **kwargs: str) -> str:
        if name is not None and is_url(name):
            # Specify the downloading link directly. "kwargs" are ignored in this case.
            url = name
        else:
            url = self.get_url(name=name, version=version, **kwargs)

        outdir = self.cachedir / str_to_hash(url)
        filename = self._get_file_name(url)
        # Download the model file if not existing
        if not (outdir / filename).exists():
            download(url, outdir / filename)

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
        self, name: str = None, version: int = -1, **kwargs: str
    ) -> Dict[str, Union[str, List[str]]]:
        if name is not None and is_url(name):
            # Specify the downloading link directly. "kwargs" are ignored in this case.
            url = name
        else:
            url = self.get_url(name=name, version=version, **kwargs)
        # Unpack to <cachedir>/<hash> in order to give an unique name
        outdir = self.cachedir / str_to_hash(url)

        # Skip downloading and unpacking if the cache exists
        meta_yaml = outdir / "meta.yaml"
        if meta_yaml.exists():
            info = get_dict_from_cache(meta_yaml)
            if info is not None:
                return info

        # Download the file to an unique path
        filename = self.download(url)

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
        "--cachedir", help="Specify cache dir. By default, download to module root.",
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
        "--key", default="name", help="The key name you want",
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="Given desired condition in form of <key>=<value>. "
        "e.g. --condition fs=16000",
    )
    parser.add_argument(
        "--cachedir", help="Specify cache dir. By default, download to module root.",
    )
    args = parser.parse_args(cmd)

    conditions = dict(s.split("=") for s in args.condition)
    d = ModelDownloader(args.cachedir)
    for v in d.query(args.key, **conditions):
        print(v)
