"""
Just some functions and objects to facilitate IO operations
"""
import codecs
import copy
import hashlib
import json
import logging
import os.path
import pickle
import time
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from utils.git_utils import get_current_commit

logger = logging.getLogger(__name__)

CURL_COMMAND = 'curl -s "{url}" -o {output_file}'


def write_json(json_dict: Dict[str, Any], filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)


def load_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        json_dict = json.load(f)
    return json_dict


def write_pickle(obj: object, output_path: str) -> None:
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def read_pickle(path: str) -> object:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def load(path: str) -> Any:
    if path.endswith(".json"):
        return load_json(path)
    elif path.endswith(".pkl"):
        return read_pickle(path)
    else:
        raise ValueError(f"{path} is not a json or pkl")


def get_pickle_str(obj: object) -> str:
    return codecs.encode(pickle.dumps(obj), "base64").decode()


def load_pickle_str(pickle_str: str) -> Any:
    return pickle.loads(codecs.decode(pickle_str.encode(), "base64"))


@dataclass
class SerializableDataclass:
    def to_json_dict(self) -> Dict[str, Any]:
        pickle_str = get_pickle_str(self)
        json_dict = copy.deepcopy(vars(self))
        for key, value in json_dict.items():
            if hasattr(value, "to_json_dict"):
                value = value.to_json_dict()
            elif isinstance(value, Enum):
                value = str(value)
            json_dict[key] = value
        json_dict["pickle_str"] = pickle_str
        json_dict["commit_hash"] = get_current_commit()
        return json_dict

    @staticmethod
    def from_json_dict(json_dict: Dict[str, Any]) -> Any:
        return load_pickle_str(json_dict["pickle_str"])

    @classmethod
    def from_path(cls, filepath: str) -> Any:
        return cls.from_json_dict(load(filepath))


def download_url(url: str, output_filename: str) -> None:
    urllib.request.urlretrieve(url, output_filename)


def curl_download(url: str, output_file: str, tries_left: int = 5) -> None:
    os.system(CURL_COMMAND.format(url=url, output_file=output_file))
    try:
        assert os.path.exists(output_file)
    except AssertionError as _:
        if tries_left <= 0:
            raise FileNotFoundError
        logger.warning(
            f"CURL: {url} failed to download at {output_file} | {tries_left} tries remaining."
        )
        time.sleep(5 * 2 ** (1 / tries_left))
        curl_download(url, output_file, tries_left=tries_left - 1)


def compute_sha256_checksum(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def file_compression(filepath: str) -> Optional[str]:
    if filepath.endswith(".gz"):
        return "gzip"
    if filepath.endswith(".zip"):
        return "zip"

    return None
