# Generatd with CHatGPT on 2021-08-25 15:00:00 https://chat.openai.com/share/133de26b-e5f5-4af8-a990-4a2b19d02254
from datetime import datetime
import importlib
import json
import logging
from os import getenv
import os
from llama_index.indices.base import IndexType
from llama_index.storage import StorageContext
from llama_index import Document, VectorStoreIndex
from llama_index.vector_stores import (OpensearchVectorClient,
                                       OpensearchVectorStore)
from llama_index.vector_stores.opensearch import (OpensearchVectorClient,
                                                  OpensearchVectorStore)
from pandarallel import pandarallel
from pandas import DataFrame, concat, json_normalize, read_parquet
import hashlib
import duckdb
import yaml


LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()

logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


class DotAccessibleDict(dict):
    def __getattr__(self, name):
        return self[name]


# Read the configuration file
with open(f"{os.path.dirname(os.path.abspath(__file__))}/../../config/config.yaml", 'r') as file:
    yaml_content = yaml.safe_load(file)

config = DotAccessibleDict(yaml_content)

pandarallel.initialize(progress_bar=True) # , nb_workers=3)


def get_string_before_last_dot(text):
    if '.' in text:
        return text.rsplit('.', 1)[0]
    else:
        return None


def dynamic_load_function(full_function_name: str):
    # Dynamically import the module
    module_name = get_string_before_last_dot(full_function_name)
    logger.info(f"Module name: {module_name}")
    if module_name:
        module = importlib.import_module(module_name)
        func_name = full_function_name.split('.')[-1]
        logger.info(f"Function name: {func_name}")
        func = getattr(module, func_name, None)
    else:
        logger.info(f"Function name: {full_function_name}")
        func = globals()[full_function_name]
    return func


def dynamic_load_function_from_env_varaibale_or_config(key: str):
    full_function_name = getenv(key) or config.function.get(key)
    if not full_function_name:
        raise ValueError(
            f"Function name not found in environment variable {key} or config file.")
    return dynamic_load_function(full_function_name)


augment_function = dynamic_load_function_from_env_varaibale_or_config(
    'augment_function')
pre_screening_function = dynamic_load_function_from_env_varaibale_or_config(
    'pre_screening_function')
screening_function = dynamic_load_function_from_env_varaibale_or_config(
    'screening_function')
embedding_function = dynamic_load_function_from_env_varaibale_or_config(
    'embedding_function')
filtering_function = dynamic_load_function_from_env_varaibale_or_config(
    'filtering_function')

def generate_df(df: DataFrame, error_column_name: str = 'error'):
    """
    Generates GOOD and BAD df based on a condition in a specific column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    error_column_name (str): The name of the column to check the condition.

    Returns:
    good_df (pd.DataFrame): Dataset where the specified column is not None.
    bad_df (pd.DataFrame): Dataset where the specified column is None.
    """
    # Check if the specified column exists in the DataFrame
    if error_column_name not in df.columns:
        raise ValueError(
            f"Column '{error_column_name}' not found in DataFrame.")

    # Splitting the DataFrame into GOOD and BAD df
    bad_df = df[df[error_column_name].notna() & (
        df[error_column_name].apply(lambda x: isinstance(x, str) and x != 'nan'))]
    good_df = df[df[error_column_name].isna() | (
        df[error_column_name].astype(str) == 'nan')]

    return good_df, bad_df

# Borrowed from https://docs.llamaindex.ai/en/stable/examples/vector_stores/OpensearchDemo.html


def embed_row(row: dict, column_name: str = 'text'):
    payload = {
        "input": [row[column_name]]
    }
    try:
        embeddings = embedding_function(payload=payload,
                                    inference_url=config.embedding.get(
                                        'embedding_url')
                                    )
        # Create a hash of row[column_name] as the ID
        doc_id = hashlib.sha256(row[column_name].encode()).hexdigest()
        return Document(text=row[column_name], doc_id=doc_id, embedding=embeddings[0]) if len(embeddings) > 0 else None
    except Exception as e:
        logger.error(f"Error embedding row: {repr(e)}")
        raise e

def apply_embedding(df: DataFrame, column_name: str = 'text'):
    # Apply embed_row function to each row of the DataFrame in parallel.
    # The result is a Series with lists of embeddings.
    doc_series = df.parallel_apply(embed_row, column_name=column_name, axis=1)

    # Remove None entries
    doc_series = doc_series.dropna()

    # Convert the Series to a list (array) of lists of embeddings.
    return doc_series.tolist()

def chunk_documents(documents, chunk_size):
    """ Divide documents into chunks based on length. """
    chunks = []
    current_chunk = []

    for doc in documents:
        if len(current_chunk) < chunk_size:
            current_chunk.append(doc)
        else:
            chunks.append(current_chunk)
            current_chunk = [doc]

    # Add the last chunk if it contains any documents
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def create_vector_db_collections(df: DataFrame,
                                 text_field: str = None,
                                 embedding_field: str = None,
                                 embedding_column: str = None
                                 ):
    # Code to create NOW and FUTURE collections
    # OpensearchVectorClient stores text in this field by default
    # OpensearchVectorClient stores embeddings in this field by default

   # http endpoint for your cluster (opensearch required for vector index usage)
    endpoint = getenv("OPENSEARCH_HOST") or config.embedding.get(
        "OPENSEARCH_HOST")
    # index to demonstrate the VectorStore impl
    idx = getenv("OPENSEARCH_INDEX") or config.embedding.get(
        "OPENSEARCH_INDEX")
    vector_dimension = getenv("embedding_vector_dimnesion") or config.embedding.get(
        "embedding_vector_dimnesion")
    text_field = text_field or config.embedding.get('text_field')
    embedding_field = embedding_field or config.embedding.get(
        'embedding_field')
    embedding_column = embedding_column or config.embedding.get(
        'embedding_column')
    chunk_size = config.embedding.get('chunk_size')

    # load some sample data
    documents = apply_embedding(df, column_name=embedding_column)
    if len(documents) > 0:
        # OpensearchVectorClient encapsulates logic for a
        # single opensearch index with vector search enabled
        client = OpensearchVectorClient(
            endpoint, idx, vector_dimension, embedding_field=embedding_field, text_field=text_field
        )
        # initialize vector store
        vector_store = OpensearchVectorStore(client)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        # initialize an index using our sample data and the client we just created
        document_chunks = chunk_documents(documents, chunk_size) if chunk_size else [documents]

        indexes = []
        for chunk in document_chunks:
            index = VectorStoreIndex.from_documents(
                documents=chunk, storage_context=storage_context
            )
            indexes.append(index)
        return indexes[-1]
        # index.set_index_id(f"{knowledge.id}")
        # index.storage_context.persist(persist_dir=f"{VDB_ROOT}/{knowledge.id}", vector_store_fname=vector_store_fname)

def add_vector_db_collections(df: DataFrame, vector_index: IndexType, embedding_column: str = None):
    try:
        embedding_column = embedding_column or config.embedding.get(
            'embedding_column')
        documents = df.parallel_apply(embed_row, column_name=embedding_column, axis=1).tolist()
        for d in documents:
            vector_index.insert(document = d)
    except Exception as e:
        logger.error(f"***************************  Exception: {repr(e)} *****************")
        

    

def combine_datasets(good_df, bad_df, size: int = None):
    """
    Randomly picks df with an 80% GOOD and 20% BAD ratio.

    Parameters:
    good_df (pd.DataFrame): The GOOD df.
    bad_df (pd.DataFrame): The BAD df.
    size (int): Total number of df to pick.

    Returns:
    pd.DataFrame: A DataFrame containing the randomly picked df.
    """
    # Calculating the number of GOOD and BAD df to pick
    
    size = size or config.refine_workflow.get('combined_df_size')
    
    size = min(size, len(good_df))
    
    num_good = int(size * 0.8)
    num_bad = size - num_good
    
   # Randomly sampling GOOD and BAD df
    good_sample = good_df.sample(n=num_good)
    bad_sample = bad_df.sample(n=num_bad)

    # Concatenating the samples into a single DataFrame
    result = concat([good_sample, bad_sample])

    return result


def augment_dataframe(df: DataFrame,
                      template: str = None,
                      inference_response_limit: int = 100,
                      instruction: str = None,
                      output_column: str = None
                      ):
    # Augment df 100 times with GPT call
    template = template or config.augmentation.get('prompt_template')
    inference_response_limit = inference_response_limit or config.augmentation.get(
        'inference_response_limit')
    instruction = instruction or config.augmentation.get('instruction')
    output_column = output_column or config.augmentation.get('output_column')
    inference_url = config.augmentation.get('inference_url')
    payload_template = config.augmentation.get('payload_template')
    output_separator = config.augmentation.get('output_separator')
    table_fields_mapping_file = config.screening.get('table_fields_mapping_file')

    result_df = concat(df[output_column].parallel_apply(augment_function,
                                                        template=template,
                                                        inference_response_limit=inference_response_limit,
                                                        instruction=instruction,
                                                        inference_url=inference_url,
                                                        payload_template=payload_template,
                                                        output_column=output_column,
                                                        output_separator=output_separator,
                                                        table_fields_mapping_file=table_fields_mapping_file,
                                                        ).tolist(), ignore_index=True)

    return result_df



def screeen_dataframes(df: DataFrame,
                       template: str = None,
                       inference_response_limit: int = 100,
                       instruction: str = None,
                       input_column: str = None,
                       output_column: str = None,
                       screening_column: str = None,
                       apiResultCount_column: str = None,
                       is_empty_result_acceptable: bool = False
                       ):

    # Augment df 100 times with GPT call
    template = template or config.screening.get('prompt_template')
    inference_response_limit = inference_response_limit or config.screening.get(
        'inference_response_limit')
    instruction = instruction or config.screening.get('instruction')
    input_column = input_column or config.screening.get('input_column')
    output_column = output_column or config.screening.get('output_column')
    inference_url = config.screening.get('inference_url')
    payload_template = config.screening.get('payload_template')
    output_separator = config.screening.get('output_separator')
    table_fields_mapping_file = config.screening.get('table_fields_mapping_file')
    
    # df = df[:5]
    # pandarallel.initialize(progress_bar=True) 
    screened_df = concat(df.parallel_apply(pre_screening_function,
                                                          template=template,
                                                          inference_response_limit=inference_response_limit,
                                                          instruction=instruction,
                                                          inference_url=inference_url,
                                                          payload_template=payload_template,
                                                          input_column=input_column,
                                                          output_column=output_column,
                                                          output_separator=output_separator,
                                                          filtering_function=filtering_function,
                                                        table_fields_mapping_file=table_fields_mapping_file,
                                                         axis=1 ).tolist(), ignore_index=True)

    # screening df through API call

    df = screened_df
    if df.empty:
        return df
    
    screening_column = screening_column or config.screening.get(
        'screening_column')
    apiResultCount_column = apiResultCount_column or config.screening.get(
        'api_response_item_count')
    
    # # Filtering rows where the 'sql' column starts with 'select from'
    # df = df[df[output_column].str.lower().startswith('select from')]

    df[screening_column] = df[output_column].parallel_apply(
        screening_function)

    df = flatten(df, screening_column)
    df = df[df.error.isna()] if 'error' in df.columns else df

    if not is_empty_result_acceptable and apiResultCount_column in df.columns:
        rel = duckdb.sql(
            f"select * from df where {apiResultCount_column} != '0.0'")
        df = rel.to_df()
    
    return df


def flatten(df, json_column):
    df[json_column] = df[json_column].apply(json.loads)

    flattened_data = json_normalize(df[json_column])

    df_flattened = concat([df, flattened_data], axis=1).drop(columns=[json_column])

    return df_flattened.astype(str)



def update_collections(good_results, future_collection, good_collection):
    # Update collections with good results
    pass


def remove_vectors_from_now(now_collection, num_vectors):
    # Remove vectors from NOW collection
    pass


def is_collection_size_met(collection, target_size):
    # Check if collection size meets the target
    pass


def refine_sql_model():
    # Code to re-finetune SQL generation model
    # triger an argo workflow
    pass


def main():
    now = datetime.now()
    df = read_parquet(config.refine_workflow.get('input_path'))
    # df = df[:1000]
    bad_df_length = config.refine_workflow.get('bad_df_length')
    good_df, bad_df = generate_df(df=df)
    bad_df = bad_df[:bad_df_length]

    vector_index = create_vector_db_collections(df=good_df)
    
    count = 1

    while len(bad_df) > 0 and len(good_df) < 15000:
        count += 1
        combined_df = combine_datasets(
            good_df=good_df, bad_df=bad_df)
        # combined_df = combined_df[:10]
        augmented_df = augment_dataframe(combined_df)
        good_results = screeen_dataframes(augmented_df)
        input_column = config.refine_workflow.get('input_column')
        values_to_exclude = good_results[input_column]
        bad_df = bad_df[~bad_df[input_column].isin(values_to_exclude)]
        good_df = concat([good_df, good_results])
        output_path = config.refine_workflow.get('output_path')
        good_df.to_parquet(f"{output_path}_{now}_{count}.parquet")
        

        add_vector_db_collections(df=good_results, vector_index=vector_index)

    # if is_collection_size_met(future_collection, len(now_collection) * x_percent):
    #     remove_vectors_from_now(now_collection, len(future_collection))

    # if len(good_collection) >= 15000:
    #     refine_sql_model()


    # if len(bad_df) == 0:
    #     # Add bad queries to BAD dataset until it reaches 15K
    #     pass

    # Optionally, rerun the entire process


if __name__ == "__main__":
    main()
