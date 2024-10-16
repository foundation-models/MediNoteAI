import json
import logging
import os
from collections.abc import Mapping
from functools import cache
from glob import glob

import spacy
import torch

from medinote import read_dataframe

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


def literal(*args):
    return eval(args[0])


class LazyDict(Mapping):
    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def get_lazy_value(self, key):
        _, arg = self._raw_dict.__getitem__(key)
        return arg

    def __getitem__(self, key):
        func, arg = self._raw_dict.__getitem__(key)
        return func(arg)

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)


def json_prompt_base():

    result = {}
    for file in glob(f"{os.path.dirname(__file__)}/gpt_ner_json_prompts/*_prompt.json"):
        with open(file, "r", encoding="utf-8") as read_file:
            key = file.split("/")[-1].split("_prompt")[0]
            result[key] = json.load(read_file)
    return result


def adjust_config_values(params: dict) -> None:
    for key, value in params.items():
        if any(word in key for word in ["model", "datasets"]):
            if (
                isinstance(value, str)
                and not value.startswith("/")
                and value not in ["bert-base-multilingual-cased"]
            ):
                value = f"{os.path.dirname(__file__)}/../../{value}"
                params[key] = value


@cache
def load_spacy(key):
    return spacy.load(name=key[0], disable=key[1])


@cache
def read_dataset(key):
    path = key[0]
    remove_non_parquet = key[1]
    path = path if path.startswith("/") else f"{os.path.dirname(__file__)}/../../{path}"
    files = glob(rf"{path}/*")
    for file in files:
        extension = file.split(".")[-1]
        if extension != "parquet":
            logger.warning(f"File {file} is not a parquet file")
            if remove_non_parquet:
                os.remove(file)
                logger.warning(f"Removed {file}")
            else:
                logger.warning(f"Raising exception for {file}")
                raise TypeError(f"File {file} is not a parquet file")
    return path


@cache
def read_dataframe_cached(key_value):
    return read_dataframe(name=key_value[0], input_path=key_value[1])


@cache
def get_folder(path):
    path = path if path.startswith("/") else f"{os.path.dirname(__file__)}/../../{path}"
    return path


def liveness_status_from_cuda_memory():
    if os.get("bypass_cuda_check"):
        return "Alive"
    if torch.cuda.is_available():
        # Get the current allocated memory
        allocated_memory = torch.cuda.memory_allocated()
        logger.debug(
            f"Currently allocated memory: {allocated_memory / 1024 ** 3:.2f} GB"
        )

        # Get the maximum allocated memory
        max_allocated_memory = torch.cuda.max_memory_allocated()
        logger.debug(f"Max allocated memory: {max_allocated_memory / 1024 ** 3:.2f} GB")

        # Check if the memory is full
        if allocated_memory == max_allocated_memory:
            logger.debug("CUDA memory is full.")
            return "Dead"
        else:
            logger.debug("CUDA memory is not full.")
            return "Alive"
    else:
        logger.debug("CUDA is not available.")
        return "Alive"
             
class Cache:
    caller_module_name = os.path.splitext(os.path.basename(__file__))[0]
    caller_file_path = os.path.dirname(__file__)


# load all class variables
Cache()
