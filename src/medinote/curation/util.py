from typing import Any
from llama_index import SimpleDirectoryReader
from sentence_splitter import SentenceSplitter, split_text_into_sentences
import os
import pandas as pd
import requests
from pandarallel import pandarallel
import logging

pandarallel.initialize(progress_bar=True) 

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def fetch_url(base_url: str, 
              row: dict, 
              text_column: str = 'text', 
              method: str = "post", 
              payload: dict = None, 
              headers: dict = None, 
              token: str = None):
    try:
        headers = headers or {
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = payload or {
            "echo": False,
            "stop": [
                "<|im_start|>"
            ],
            "prompt": "<|im_start|>system\nYou are \"Hermes 2\", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>\n<|im_start|>user\nCreate a very short, couple of words note for the following text: \n\n{{row[text_column]}}<|im_end|>\n<|im_start|>assistant\n"
        }

        if method.lower() == "post":
            response = requests.post(url=url, headers=headers, json=payload)
        else:
            url = f"{base_url}/{row[text_column]}"
            response = requests.get(url=url, headers=headers)
        return response.text  # or response.json() based on the response type
    except requests.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        raise  # Re-raise the exception after logging

def fetch_and_save_data(start_index: int = None, df_length: int = None):
    # Initialize pandarallel
    pandarallel.initialize(progress_bar=True)

    # Read environment variables
    base_url = os.environ.get('BASE_CURATION_URL', 'https://example.com')  # Default if not set
    source_path = os.environ['SOURCE_DATAFRAME_PARQUET_PATH']
    output_path = os.environ['OUTPUT_DATAFRAME_PARQUET_PATH']
    start_index = os.environ.get('START_INDEX', None)
    df_length = os.environ.get('DF_LENGTH', None)


    # Read the DataFrame from Parquet file
    df = pd.read_parquet(source_path)
    if start_index is not None:
        df = df[int(start_index):]
    if df_length is not None:
        df = df[:int(df_length)]
        
    # Apply the function in parallel to the 'text' column
    df['result'] = df['text'].parallel_apply(fetch_url, args=(base_url,))

    # Save the modified DataFrame to a Parquet file
    df.to_parquet(output_path)


def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")
    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)
    if verbose:
        print(f"Parsed {len(nodes)} nodes")
    return nodes

def update_message_bad(function_dict: dict, message: dict):
    for function in function_dict.values():
        globals_dict = {"my_function": function}
        locals_dict = {"message": message}

        exec("result = my_function(message)", globals_dict, locals_dict)
        returned_message = locals_dict["result"]
        if returned_message:
            return returned_message
    return message


def update_message(function_dict: dict, input: dict, agent: Any, samples: list = None):
    for function in function_dict.values():
        locals_dict = {"input": input, "samples": samples}
        exec(function, {}, locals_dict)
        returned_message = locals_dict.get("result")
        if returned_message:
            agent.update_system_message('')
            return returned_message
    return input

if __name__ == "__main__":
    fetch_and_save_data()