# Generatd with CHatGPT on 2021-08-25 15:00:00 https://chat.openai.com/share/133de26b-e5f5-4af8-a990-4a2b19d02254
import importlib
import json
import logging
import os
from pandarallel import pandarallel
from pandas import concat, json_normalize
import yaml
import glob
import pandas as pd


class DotAccessibleDict(dict):
    def __getattr__(self, name):
        return self[name]


def initialize():
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
    logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
    logger.addHandler(logging.FileHandler(
        f"{os.path.dirname(os.path.abspath(__file__))}/logs/{os.path.splitext(os.path.basename(__file__))[0]}.log"))

    # Add logger format to show the name of file and date time before the log statement
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Set up error formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s - Line %(lineno)d')

    # Apply error formatter to handlers
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(formatter)

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


def flatten(df, json_column):
    df[json_column] = df[json_column].apply(json.loads)

    flattened_data = json_normalize(df[json_column])

    df_flattened = concat([df, flattened_data], axis=1).drop(
        columns=[json_column])

    return df_flattened.astype(str)

def merge_parquet_files(pattern):
    file_list = glob.glob(pattern)
    df = pd.concat([pd.read_parquet(file) for file in file_list])
    return df