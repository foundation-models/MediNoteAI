

from glob import glob
import os
import duckdb
from pandas import DataFrame, Series, concat, merge, read_parquet
from medinote import dynamic_load_function_from_env_varaibale_or_config, initialize, merge_parquet_files
from medinote.augmentation.datafarme_search import search_df
from medinote.augmentation.sql_based_augmentation import generate_sql_schema
from medinote.curation.rest_clients import generate_via_rest_client
from medinote.augmentation.sql_based_augmentation import generate_sql_schema


config, logger = initialize()

get_fields_from_obj_name_function = dynamic_load_function_from_env_varaibale_or_config("get_fields_from_obj_name_function") 
list_obj_names_function = dynamic_load_function_from_env_varaibale_or_config("list_obj_names_function")
""" 
develoging based on this plan: 
https://chat.openai.com/share/b5cc5846-141a-4b57-8560-8065236552d8
"""

def generate_schema_df(obj_name: str = None):
    """_summary_

    Generate a schema DataFrame based on a specified object name.
    """
    obj_name, fields = get_fields_from_obj_name_function(obj_name)
    schema = generate_sql_schema(obj_name, fields)
    df = DataFrame(columns=['obj_name', 'schema', 'field', 'type'])
    for field in fields:
        new_row = {'obj_name': obj_name, 'schema': schema, 'field': field[0], 'type': field[1]}
        df = concat([df, DataFrame(new_row, index=[0])], ignore_index=True)
    return df

def generate_synthetic_data(row: Series, 
                            obj_name: str = None,
                            schema_df: DataFrame = None
                            ):
    """_summary_

    Generate synthetic data based on a specified object name.
    """
    try:
        if schema_df is None and obj_name:
            schema_df = generate_schema_df(obj_name)
        elif obj_name is None and schema_df is None:
            raise ValueError(f"neither schema_df nor obj_name is provided")
            
        input_column_name = config.sqlcoder.get("input_column_name") or "text"
        question = row[input_column_name]
        
        if not obj_name:
            obj_name_column = config.sqlcoder.get("obj_name_column") or "obj_name"
            obj_name = row[obj_name_column]
        
        sql_schema = schema_df[schema_df['obj_name'].str.lower() == obj_name]['schema'].values[0]
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
    except Exception as e:
        logger.error(f"Error generating synthetic data: {repr(e)}")
        return row


def parallel_generate_synthetic_data(obj_name: str = None, 
                                     df: DataFrame = None, 
                                     df_processed: DataFrame = None
                                     ):
    """_summary_

    Generate synthetic data based on a specified object name.
    """
    # if obj_name:
    #     df = search_df(obj_name, df=df)
    input_path = config.sqlcoder.get("input_path")
    if df is None and input_path:
        logger.debug(f"Reading the input parquet file from {input_path}")
        df = read_parquet(input_path)
        
    if obj_name:
        obj_name_column = config.sqlcoder.get("obj_name_column") or "obj_name"
        df = df[df[obj_name_column] == obj_name]
        schema_df = generate_schema_df(obj_name)
    else:
        schema_df = None
    text_column_name = config.sqlcoder.get("text_column_name") or "text"
    
    processed_path = config.sqlcoder.get("processed_path")
    
    if df_processed is None and processed_path:
        logger.debug(f"Reading the processed parquet file from {processed_path}")
        df_processed = read_parquet(processed_path)
        
    if df_processed is not None:
        # Perform a left merge with an indicator
        merged = merge(df, df_processed[[text_column_name]], on=text_column_name, how='left', indicator=True)

        # Filter rows where '_merge' is 'left_only'
        df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
            
    output_path = config.sqlcoder.get("output_path")
    
    chunk_size = 10
    num_chunks = len(df) // chunk_size + 1
    
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df))
        chunk_df = df[start_index:end_index]
        
        output_file = f"{output_path}_{obj_name}_{start_index}_{end_index}.parquet" if output_path else None
        if output_file is None or not os.path.exists(output_file):      
            try:
                chunk_df = chunk_df.parallel_apply(generate_synthetic_data, axis=1, schema_df=schema_df, obj_name=obj_name)
            except ValueError as e:
                if "Number of processes must be at least 1" in str(e):
                    logger.error(f"No idea for error: Number of processes must be at least \n ignoring .....")
            except Exception as e:
                logger.error(f"Error generating synthetic data: {repr(e)}")
                
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


def merge_all_embedding_files(pattern: str = None, 
                             output_path: str = None):
    pattern = pattern or config.sqlcoder.get('merge_pattern')
    output_path = output_path or config.sqlcoder.get('merge_output_path')
    df = merge_parquet_files(pattern, identifier_index=2)
    df.to_parquet(output_path)
    
        
if __name__ == "__main__":
    sql_generate_for_all_objects()
