# Generatd with CHatGPT on 2021-08-25 15:00:00 https://chat.openai.com/share/133de26b-e5f5-4af8-a990-4a2b19d02254
import importlib
import json
import logging
import os
from pathlib import Path
from pandarallel import pandarallel
from pandas import DataFrame, concat, json_normalize, read_parquet
import yaml
from functools import cache
from glob import glob

from fastchat.utils import build_logger
from medinote.cached import Cache
import tracemalloc


def setup_logging(worker_id: str = None):
    """Sets up logging for each worker with a unique log file."""
    if worker_id:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            filename=f"worker_{worker_id}.log",
            filemode="a",
            format=log_format,
            level=logging.INFO,
        )
        return None
    else:
        caller_module_name = Cache.caller_module_name or "default"
        caller_file_path = Cache.caller_file_path or os.path.dirname(
            os.path.abspath(__file__)
        )
        log_path = f"{caller_file_path}/logs"
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logger_filename = f"{caller_file_path}/logs/{caller_module_name}.log"
        logger = build_logger(
            logger_name=caller_module_name, logger_filename=logger_filename
        )
        return logger


def initialize_worker():
    """Initializes each worker."""
    from dask.distributed import get_worker

    worker = get_worker()
    setup_logging(worker.id)


@cache
def initialize():
    tracemalloc.start()

    # Read the configuration file
    with open(
        f"{os.path.dirname(os.path.abspath(__file__))}/config/config.yaml", "r"
    ) as file:
        yaml_content = yaml.safe_load(file)

    config = yaml_content

    if os.getenv("USE_DASK", "False") == "False":
        if config.get("debug") or os.getenv("single_process"):
            pandarallel.initialize(progress_bar=False, nb_workers=1)
        elif config.get("pandarallel") and config.get("pandarallel").get("nb_workers"):
            pandarallel.initialize(
                progress_bar=True,
                nb_workers=config.get("pandarallel").get("nb_workers"),
            )
        else:
            pandarallel.initialize(progress_bar=True)

    return config, setup_logging()


def get_string_before_last_dot(text):
    if "." in text:
        return text.rsplit(".", 1)[0]
    else:
        return None


def dynamic_load_function(full_function_name: str):
    # Dynamically import the module
    module_name = get_string_before_last_dot(full_function_name)
    if module_name:
        module = importlib.import_module(module_name)
        func_name = full_function_name.split(".")[-1]
        func = getattr(module, func_name, None)
    else:
        func = globals()[full_function_name]
    return func


def dynamic_load_function_from_env_varaibale_or_config(key: str, config: dict):

    full_function_name = os.getenv(key) or config.get("function").get(key)
    if not full_function_name:
        raise ValueError(
            f"Function name not found in environment variable {key} or config file."
        )
    return dynamic_load_function(full_function_name)


def flatten(
    df: DataFrame,
    json_column: str,
):
    df[json_column] = df[json_column].apply(json.loads)
    df[json_column] = df[json_column].apply(
        lambda row: json.loads(row) if isinstance(row, str) else row
    )

    flattened_data = json_normalize(df[json_column])

    df = df.join(flattened_data)

    return df.astype(str)


def merge_parquet_files(
    pattern: str,
    identifier_delimiter: str = "_",
    identifier_index: int = -1,
    identifier: str = None,
):
    file_list = glob(pattern)
    # df = concat([read_parquet(file) for file in file_list]) if identifier_index == -1 else concat(
    #     [read_parquet(file).assign(identifier=file.split(identifier_delimiter)[identifier_index]) for file in file_list])

    # Modified list comprehension with error handling
    dataframes = []
    if not file_list:
        raise ValueError(f"No files found with pattern {pattern}")
    for file in file_list:
        try:
            if identifier:
                df = read_parquet(file).assign(identifier=identifier)
            elif identifier_index != -1:
                df = read_parquet(file).assign(
                    identifier=file.split(identifier_delimiter)[identifier_index]
                )
            else:
                df = read_parquet(file)
            dataframes.append(df)
        except Exception as e:
            logging.error(f"Failed to read {file}: {e}")
            Path(f"{file}.not_merged").touch()

    # Concatenate the successfully read dataframes
    concatenated_df = concat(dataframes)
    return concatenated_df
