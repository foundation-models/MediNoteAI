import os
from typing import Any

from pandas import DataFrame
from llama_index import SimpleDirectoryReader
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from pandarallel import pandarallel
import logging


pandarallel.initialize(progress_bar=True) 

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


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

instruction_prompt = os.environ.get('INSTRUCTION_PROMPT', 'Response based on the following pattern')
input_column = os.environ.get('INPUT_COLUMN', 'input')
output_column = os.environ.get('OUTPUT_COLUMN', 'output')

def instruction_based_prompt(row: dict):
    return f"{instruction_prompt} \n\n ### INPUT:\n {row[input_column]} \n\n ### INPUT:\n {row[output_column]}  \n\n ### END"

def generate_training_dataset(df: DataFrame, output_file: str):
    sort_column = os.environ.get('SORT_COLUMN')
    
    df['text'] = df.parallel_apply(instruction_based_prompt, axis=1)
    sorted_df = df.sort_values(by=sort_column) if sort_column else df
    sorted_df[["text"]].to_json(output_file, orient="records", lines=True)
