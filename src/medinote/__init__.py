# Generatd with CHatGPT on 2021-08-25 15:00:00 https://chat.openai.com/share/133de26b-e5f5-4af8-a990-4a2b19d02254
import importlib
import json
import logging
import os
from pathlib import Path
from pandas import DataFrame, concat, json_normalize, read_csv, read_excel, read_parquet
import yaml
from functools import cache
from glob import glob
from collections.abc import Iterable

import tracemalloc

from medinote.utils import build_logger

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


def setup_logging(worker_id: str = None, logger_name: str = None, log_path: str = None):
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
        logger_name = logger_name or os.path.splitext(os.path.basename(__file__))[0]
        caller_file_path = log_path or os.path.dirname(__file__)
        log_path = f"{caller_file_path}"
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logger_filename = f"{caller_file_path}/{logger_name}.log"
        logger = build_logger(logger_name=logger_name, logger_filename=logger_filename)
        return logger


def initialize_worker():
    """Initializes each worker."""
    from dask.distributed import get_worker

    worker = get_worker()
    setup_logging(worker.id)


@cache
def initialize(logger_name: str = None, root_path: str = None):
    tracemalloc.start()
    root_path = root_path or f"{os.path.dirname(__file__)}"

    # Read the configuration file
    try:
        with open(f"{root_path}/config/config.yaml", "r") as file:
            yaml_content = yaml.safe_load(file)

        config = yaml_content
    except Exception as e:
        logger.error(f"Ignoring not finding config file from: {root_path} with error: {e}")
        config = {}

    if os.getenv("USE_PANDARALLEL", "true").lower() == "true":
        from pandarallel import pandarallel

        if nb_workers := os.getenv("nb_workers"):
            logger.info(f"Using {nb_workers} workers for pandarallel")
            pandarallel.initialize(progress_bar=nb_workers != 1, nb_workers=int(nb_workers))
        elif config.get("pandarallel") and config.get("pandarallel").get("nb_workers"):
            pandarallel.initialize(
                progress_bar=True,
                nb_workers=int(config.get("pandarallel").get("nb_workers")),
            )
        else:
            pandarallel.initialize(progress_bar=True)

    return config, setup_logging(logger_name=logger_name, log_path=f"{root_path}/logs")


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


def dynamic_load_function_from_env_varaibale_or_config(
    key: str, config: dict, default_function: str = None
):

    full_function_name = (
        os.getenv(key) or config.get("function").get(key)
        if config.get("function")
        else default_function
    )
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


def read_any_dataframe(dataframe_name: str):
    any_dataframe = f"**/*/{dataframe_name}"
    logger.info("searching for {any_dataframe}")
    return read_dataframe(input_path=any_dataframe, ignore_not_supported=True)


def read_df_from_json(file):
    with open(file, "r") as f:
        data = json.load(f)
        df = json_normalize(data)
        return df


def read_dataframe(
    input_path: str,
    input_header_names: str = None,
    max_records_to_process: int = -1,
    header: int = 0,
    ignore_not_supported: bool = False,
    name: str = None,
    do_create_dataframe: bool = False,
    csv_names: list = None,
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
        if extension in ["json", "jsonl"]:
            df = read_df_from_json(file)
        elif extension in ["parquet"]:
            df = read_parquet(file)
        elif extension in ["csv"]:
            df = read_csv(file, header=header, names=csv_names)
        elif extension in ["txt", "text"]:
            # Read the file line-by-line
            with open(file, "r") as file:
                lines = file.readlines()

            # Remove newline characters from each line
            lines = [line.strip() for line in lines]

            # Create a DataFrame with one column named 'text'
            df = DataFrame(lines, columns=["text"])
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


# Function to check if a column has mixed types
def has_mixed_types(col):
    dtype = None
    for val in col:
        if dtype is None:
            dtype = type(val)
        elif type(val) != dtype:
            return True
    return False


def fix_prablems_with_columns(df):
    # Let's iterate over each column and try to convert it to Parquet format individually
    problematic_columns = []

    for column in df.columns:
        temp_df = DataFrame(df[column])
        try:
            # Try converting each column to Parquet format
            temp_df.to_parquet(f"/tmp/{column}.parquet")
        except Exception as e:
            logger.error(f"Error with column {column}: {str(e)}")
            # If there's an error, we add it to the list of problematic columns
            problematic_columns.append(column)

    # Now convert problematic columns to string
    for column in problematic_columns:
        df[column] = df[column].astype(str)
    return df


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
            for col in df.columns:
                if has_mixed_types(df[col]):
                    df[col] = df[col].astype(str)
            try:
                df.to_parquet(output_path)
            except Exception as e:
                logger.warning(
                    f"Error saving to {output_path}: {repr(e)} \n retrying ..."
                )
                df = fix_prablems_with_columns(df=df)
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


def remove_files_with_pattern(pattern: str):
    file_list = glob(pattern)
    for file in file_list:
        os.remove(file)


def merge_all_chunks(
    pattern: str,
    output_path: str,
    obj_name: str = None,
    column_names_map: dict = None,
    remove_pattern_matched_files: bool = False,
):
    df = merge_parquet_files(pattern, identifier=obj_name)
    logger.info(f"Merging all Screening files to {output_path}")
    if column_names_map:
        df.rename(columns=column_names_map, inplace=True)
    write_dataframe(df=df, output_path=output_path)
    if remove_pattern_matched_files:
        remove_files_with_pattern(pattern)
    return df


def chunk_process(
    function: callable,
    df: DataFrame = None,
    chunk_size: int = None,
    config: dict = None,
    persist: bool = True,
    complimentary_df: DataFrame = None,
):
    internal_logger = config.get("logger") or logger
    input_path = config.get("input_path")
    if df is None and input_path:
        df = read_dataframe(input_path)

    if df_n_samples := config.get("df_n_samples"):
        df = df.sample(n=df_n_samples)

    if selected_columns := config.get("selected_columns"):
        df = df[selected_columns]

    if df_query := config.get("df_query"):
        df_filtered = df.query(df_query)
    else:
        df_filtered = df

    if column_names_map := config.get("column_names_map"):
        df.rename(columns=column_names_map, inplace=True)

    chunk_size = chunk_size or config.get("chunk_size") or 1000

    num_chunks = (
        len(df_filtered) // chunk_size + 1 if chunk_size and chunk_size > 0 else 0
    )
    internal_logger.info(
        f"Processing {len(df_filtered)} rows in {num_chunks} chunks of size {chunk_size}"
    )
    output_prefix = (
        config.get("output_prefix")
        or get_string_before_last_dot(config.get("input_path")) + "_chunks"
    )

    chunk_df_list = []
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df_filtered))
        chunk_df = df_filtered[start_index:end_index]
        output_chunk_file = (
            f"{output_prefix}_{start_index}_{end_index}.parquet"
            if output_prefix
            else None
        )

        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(output_chunk_file)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        if function and not os.path.exists(output_chunk_file):
            try:
                chunk_df = (
                    chunk_df.parallel_apply(
                        function,
                        axis=1,
                        config=config,
                        complimentary_df=complimentary_df,
                    )
                    if complimentary_df is not None
                    else chunk_df.parallel_apply(
                        function,
                        axis=1,
                        config=config,
                    )
                )
                internal_logger.info(f"Processed chunk {chunk_df.shape} rows.")
                if not isinstance(chunk_df, DataFrame) and isinstance(
                    chunk_df, Iterable
                ):
                    chunk_df = concat(chunk_df.values, ignore_index=True)
                chunk_df_list.append(chunk_df)
            except ValueError as e:
                if "Number of processes must be at least 1" in str(e):
                    internal_logger.error(
                        f"Probably chunk_df is empty: Number of processes must be at least \n ignoring ....."
                    )
            except Exception as e:
                internal_logger.error(f"Error generating synthetic data: {repr(e)}")

            if persist:
                try:
                    write_dataframe(chunk_df, output_chunk_file)
                except Exception as e:
                    internal_logger.error(f"Error saving to {output_chunk_file}: {repr(e)}")
        else:
            internal_logger.info(
                f"Skipping chunk {start_index} to {end_index} as it already exists."
            )
    if persist:
        pattern = f"{output_prefix}_*.parquet"
        output_path = config.get("output_path") or f"{output_prefix}.parquet"
        if num_chunks > 0:
            merged_df = merge_all_chunks(
                pattern=pattern,
                output_path=output_path,
                column_names_map=config.get("column_names_map"),
            )
        else:
            output_path = config.get("output_path")
            if output_path:
                write_dataframe(df, output_path)
            return df
    else:
        merged_df = concat(chunk_df_list, ignore_index=True)
    return merged_df


# Function to flatten the JSON column
def flatten_json(df, json_column):
    # Convert JSON strings to dictionaries
    df[json_column] = df[json_column].apply(lambda x: json.loads(x))

    # Normalize JSON column
    df_normalized = json_normalize(df[json_column])

    # Concatenate the normalized columns with the original DataFrame
    df = concat([df, df_normalized], axis=1)

    return df.drop(columns=[json_column])
