

import os
import duckdb
from pandas import DataFrame, Series, read_parquet
from medinote import dynamic_load_function_from_env_varaibale_or_config, initialize, merge_parquet_files
from medinote.augmentation.datafarme_search import search_df
from medinote.augmentation.sql_based_augmentation import generate_sql_schema
from medinote.curation.rest_clients import generate_via_rest_client


config, logger = initialize()

get_fields_from_obj_name_function = dynamic_load_function_from_env_varaibale_or_config("get_fields_from_obj_name_function") 
list_obj_names_function = dynamic_load_function_from_env_varaibale_or_config("list_obj_names_function")
""" 
develoging based on this plan: 
https://chat.openai.com/share/b5cc5846-141a-4b57-8560-8065236552d8
"""



def generate_synthetic_data(row: Series, obj_name: str = None):
    """_summary_

    Generate synthetic data based on a specified object name.
    """
    obj_name, fields = get_fields_from_obj_name_function(obj_name)
    sql_schema = generate_sql_schema(obj_name, fields)

    input_column_name = config.sqlcoder.get("input_column_name") or "text"
    question = row[input_column_name]
    row_dict = {'question': question, 'ddl': sql_schema}

    template = config.sqlcoder['prompt_template']
    logger.debug(f"Using template: {template}")
    prompt = template.format(**row_dict)

    prompt_column = config.sqlcoder['prompt_column']
    if prompt_column:
        row[prompt_column] = prompt

    template = config.sqlcoder.get('payload_template')
    payload = template.format(**{"prompt": prompt})

    inference_url = config.inference.get('inference_url')
    response = generate_via_rest_client(payload=payload,
                                        inference_url=inference_url
                                        )
    output_column_name = config.sqlcoder.get("output_column_name") or "inference"

    row[output_column_name] = response.replace('\n', ' ').strip()

    return row


def parallel_generate_synthetic_data(obj_name: str = None):
    """_summary_

    Generate synthetic data based on a specified object name.
    """
    df = search_df(obj_name)
    output_path = config.sqlcoder.get("output_path")
    
    chunk_size = 10
    num_chunks = len(df) // chunk_size + 1
    
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df))
        chunk_df = df[start_index:end_index]
        
        output_file = f"{output_path}_{obj_name}_{start_index}_{end_index}.parquet" if output_path else None
        if output_file is None or not os.path.exists(output_file):      
            chunk_df = chunk_df.parallel_apply(generate_synthetic_data, axis=1, obj_name=obj_name)

            if output_file:
                try:
                    chunk_df.to_parquet(output_file)
                except Exception as e:
                    logger.error(
                        f"Error saving the embeddings to {output_file}: {repr(e)}")
        else:
            logger.info(
                f"Skipping chunk {start_index} to {end_index} as it already exists.")



def sql_generate_for_all_objects():
    objec_names = list_obj_names_function()
    for obj_name in objec_names:
        parallel_generate_synthetic_data(obj_name)
        
if __name__ == "__main__":
    sql_generate_for_all_objects()
