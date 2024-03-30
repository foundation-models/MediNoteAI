import json
import logging
import os
from collections.abc import Mapping
from functools import cache
from glob import glob

import spacy
import torch
from pandas import DataFrame, concat, json_normalize, read_csv, read_excel, read_parquet

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


def read_df_from_json(file):
    with open(file, "r") as f:
        data = json.load(f)
        df = json_normalize(data)
        return df


@cache
def read_dataframe_cached(key_value):
    return read_dataframe(name=key_value[0], input_path=key_value[1])


def read_any_dataframe(dataframe_name: str):
    any_dataframe = f"**/*/{dataframe_name}"
    logger.info("searching for {any_dataframe}")
    return read_dataframe(input_path=any_dataframe, ignore_not_supported=True)


def read_dataframe(
    input_path: str,
    input_header_names: str = None,
    max_records_to_process: int = -1,
    header: int = 0,
    ignore_not_supported: bool = False,
    name: str = None,
    do_create_dataframe: bool = False,
    on_bad_lines="error",
    encoding_errors="strict",
):
    df_all = DataFrame()
    df = DataFrame()
    input_path = (
        input_path
        if input_path.startswith("/")
        else f"{os.path.dirname(__file__)}/../../{input_path}"
    )

    if do_create_dataframe and not os.path.exists(input_path):
        return DataFrame()

    files = glob(rf"{input_path}")

    for file in files:
        extension = file.split(".")[-1]
        if extension in ["dvc"]:
            pass
        if extension in ["json"]:
            df = read_df_from_json(file)
        elif extension in ["parquet"]:
            df = read_parquet(file)
        elif extension in ["csv"]:
            df = read_csv(
                file,
                header=header,
                names=input_header_names,
                on_bad_lines=on_bad_lines,
                encoding_errors=encoding_errors,
            )
        elif extension in ["tsv"]:
            df = read_csv(
                file,
                header=header,
                names=input_header_names,
                encoding_errors=encoding_errors,
                on_bad_lines=on_bad_lines,
                sep="\t",
            )
        elif extension in ["xls", "xlsx"]:
            df = read_excel(file, header=header, names=input_header_names)
        elif not ignore_not_supported:
            logger.error(f"File extension {extension} not supported")
            raise TypeError("File extension not supported")
        else:
            pass

            # df = read_fwf(file, header=None, names=[params.text_column]) if extension in ['source'] else df
        if len(df) > 0 and max_records_to_process > 0:
            max_records_to_process = max_records_to_process
            df = df[:max_records_to_process] if max_records_to_process else df
        df_all = concat([df_all, df], ignore_index=True)
    df_all.path = input_path
    df_all.name = name
    return df_all


def write_dataframe(df, output_path: str, do_concat: bool = False):
    """_summary_

    Write data to a file.
    """
    if do_concat:
        if os.path.exists(output_path):
            df = concat([read_dataframe(output_path), df], ignore_index=True)

    if output_path:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        if output_path.endswith(".parquet"):
            df["dropped_feature_ids"] = df["dropped_feature_ids"].astype(str)
            df.to_parquet(output_path)
        elif output_path.endswith(".csv"):
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(
                f"Unsupported file format for {output_path}. Only .parquet and .csv are supported."
            )
        logger.info(f"Written {len(df)} rows to {output_path}")
    else:
        raise ValueError(f"output_path is not provided")


# Function to get parent folder name
def get_parent_folder_name(file_path):
    return os.path.basename(os.path.dirname(file_path))


# Function to get file name
def get_file_name(file_path):
    return os.path.basename(file_path)


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
