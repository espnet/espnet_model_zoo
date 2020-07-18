import hashlib
from pathlib import Path
import re
import shutil
import tempfile
from typing import Dict
from typing import List
from typing import Union

import pandas as pd
import requests
from tqdm import tqdm

from espnet2.main_funcs.pack_funcs import get_dict_from_cache
from espnet2.main_funcs.pack_funcs import unpack


MODELS_URL = (
    "https://raw.githubusercontent.com/espnet/espnet_models/master/"
    "espnet_models/table.csv"
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
    with tempfile.TemporaryDirectory() as d, (Path(d) / "tmp").open("wb") as f:
        with tqdm(
            desc=url, total=file_size, unit="B", unit_scale=True, unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        shutil.move(Path(d) / "tmp", output_path)


class ModelDownloader:
    """Download model from zenodo and unpack

    Examples:
        >>> d = ModelDownloader("~/.cache/espnet")  # Specify cachedir
        >>> d = ModelDownloader()  # <module_dir> is used as cachedir by default

        To download and unpack a model file,
        you need to input the model name which you want.

        >>> d.get_model("user_name/model_name")
        {"asr/config.yaml": <config path>, "asr/pretrain.pth": <model path>, ...}

        Instead of using the model name, you can input the url directly

        >>> d.get_model("https://zenodo.org/record/...")

        You can also get a model with specifying some attributes.

        >>> d.get_model(task="asr", corpus="wsj")

        If multiple models are matched with the condition, latest model is selected.
        You can also specify the model with "version" option.

        >>> d.get_model(task="asr", corpus="wsj", version=-1)  # Get the latest model
        >>> d.get_model(task="asr", corpus="wsj", version=-2)  # Get previous model

        You can view uploaded model names from our Zenodo community,
        https://zenodo.org/communities/espnet/

        or,

        >>> d.get_model_names()
        [...]

        You can also show them with specifying some attributes

        >>> d.get_model_names(task="asr")
        [...]

    """

    def __init__(self, cachedir: Union[Path, str] = None):
        if cachedir is None:
            cachedir = Path(__file__).parent
        else:
            cachedir = Path(cachedir).expanduser().absolute()
        cachedir.mkdir(parents=True, exist_ok=True)

        if not (cachedir / "table.csv").exists():
            download(MODELS_URL, cachedir / "table.csv")

        self.cachedir = cachedir
        self.csv = cachedir / "table.csv"
        self.data_frame = pd.read_csv(cachedir / "table.csv")

    def update_model_table(self):
        download(MODELS_URL, self.cachedir / "table.csv")

    def get_model_names(self, **kwargs) -> List[str]:
        conditions = None
        for key, value in kwargs.items():
            condition = self.data_frame[key] == str(value)
            if conditions is None:
                conditions = condition
            else:
                conditions &= condition

        if conditions is not None:
            df = self.data_frame[conditions]
            if len(df) == 0:
                return []
            else:
                return list(df["name"])
        else:
            return list(self.data_frame["name"])

    def get_model(
        self, name: str = None, version: int = -1, **kwargs: str
    ) -> Dict[str, Union[str, List[str]]]:
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
                condition = self.data_frame[key] == str(value)
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

        # URL e.g.
        # https://sandbox.zenodo.org/record/646767/files/asr_train_raw_bpe_valid.acc.best.zip?download=1
        filename = re.match(r"https://.*/([^/]*)\?download=[0-9]*$", url).groups()[0]
        # Unpack to <cachedir>/<hash> in order to give an unique name
        outdir = self.cachedir / str_to_hash(url)

        # Skip downloading and unpacking if the cache exists
        meta_yaml = outdir / Path(filename).stem / "meta.yaml"
        if meta_yaml.exists():
            info = get_dict_from_cache(meta_yaml)
            if info is not None:
                return info

        # Download the model file if not existing
        if not (self.cachedir / filename).exists():
            download(url, self.cachedir / filename)

            # MD5 checksum
            sig = hashlib.md5()
            chunk_size = 8192
            with open(self.cachedir / filename, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if len(chunk) == 0:
                        break
                    sig.update(chunk)

            checksum = list(self.data_frame[self.data_frame["url"] == url]["checksum"])[
                0
            ]
            if sig.hexdigest() != checksum:
                Path(self.cachedir / filename).unlink()
                raise RuntimeError(f"Failed to download file: {url}")

        return unpack(self.cachedir / filename, outdir)


if __name__ == "__main__":
    d = ModelDownloader()
    d.get_model_names()
    print(d.get_model("kamo-naoyuki/for_test"))
