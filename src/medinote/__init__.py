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

from medinote.cached import Cache


class DotAccessibleDict(dict):
    def __getattr__(self, name):
        return self[name]

@cache
def initialize():

    # # If logger is already configured, return it
    # if is_logger_configured:
    #     logger = logging.getLogger()
    #     logger.info(f"reusing existing logger {logger} on {logger.handlers}")
    # else:
        # Configure the logger

    caller_module_name = Cache.caller_module_name or "default"
    caller_file_path = Cache.caller_file_path or os.path.dirname(os.path.abspath(__file__))
    log_path = f"{caller_file_path}/logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)        
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
    logger = logging.getLogger(caller_module_name)
    handler = logging.FileHandler(
        f"{caller_file_path}/logs/{caller_module_name}.log")
    logger.addHandler(handler)

    # Add logger format to show the name of file and date time before the log statement
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Set up error formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s - Line %(lineno)d')

    # Apply error formatter to handlers
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(formatter)
            
    other_logger = logging.getLogger('weaviate')
    other_logger.addHandler(handler)  
    other_logger = logging.getLogger('httpx')
    other_logger.addHandler(handler)  
    other_logger = logging.getLogger('Pandarallel')
    other_logger.addHandler(handler)  
    logger.info("=========================================================================")
        # is_logger_configured = True

    # Read the configuration file
    with open(f"{os.path.dirname(os.path.abspath(__file__))}/config/config.yaml", 'r') as file:
        yaml_content = yaml.safe_load(file)

    config = DotAccessibleDict(yaml_content)

    if config['debug']:
        pandarallel.initialize(progress_bar=False, nb_workers=1)
    else:
        pandarallel.initialize(progress_bar=True)

    return config, logger


def get_string_before_last_dot(text):
    if '.' in text:
        return text.rsplit('.', 1)[0]
    else:
        return None


def dynamic_load_function(full_function_name: str):
    # Dynamically import the module
    module_name = get_string_before_last_dot(full_function_name)
    if module_name:
        module = importlib.import_module(module_name)
        func_name = full_function_name.split('.')[-1]
        func = getattr(module, func_name, None)
    else:
        func = globals()[full_function_name]
    return func


def dynamic_load_function_from_env_varaibale_or_config(key: str):
    config, logger = initialize()
    full_function_name = os.getenv(key) or config.function.get(key)
    if not full_function_name:
        raise ValueError(
            f"Function name not found in environment variable {key} or config file.")
    return dynamic_load_function(full_function_name)


def flatten(df: DataFrame,
            json_column: str,
            ):
    df[json_column] = df[json_column].apply(json.loads)
    df[json_column] = df[json_column].apply(lambda row: json.loads(row) if isinstance(row, str) else row)


    flattened_data = json_normalize(df[json_column])

    df = df.join(flattened_data)

    return df.astype(str)


def merge_parquet_files(pattern: str,
                        identifier_delimiter: str = '_',
                        identifier_index: int = -1,
                        identifier: str = None
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
                df = read_parquet(file).assign(identifier=file.split(identifier_delimiter)[identifier_index])
            else:
                df = read_parquet(file) 
            dataframes.append(df)
        except Exception as e:
            logging.error(f"Failed to read {file}: {e}")
            Path(f'{file}.not_merged').touch()

    # Concatenate the successfully read dataframes
    concatenated_df = concat(dataframes)
    return concatenated_df
