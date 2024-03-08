import os
import duckdb
from pandas import DataFrame, concat, read_parquet
from medinote import flatten, initialize, dynamic_load_function_from_env_varaibale_or_config, merge_parquet_files


config, logger = initialize()


pre_screening_function = dynamic_load_function_from_env_varaibale_or_config(
    'pre_screening_function')
screening_function = dynamic_load_function_from_env_varaibale_or_config(
    'screening_function')
filtering_function = dynamic_load_function_from_env_varaibale_or_config(
    'filtering_function')


def api_screening(df: DataFrame = None,
                  screening_column: str = None,
                  output_column: str = None,
                  apiResultCount_column: str = None,
                  is_empty_result_acceptable: bool = True
                  ):
    if df is None:
        df = read_parquet(config.screening.get('input_path'))
    output_column = output_column or config.screening.get('output_column')
    screening_column = screening_column or config.screening.get(
        'screening_column')
    apiResultCount_column = apiResultCount_column or config.screening.get(
        'api_response_item_count')
    output_path = config.screening.get('output_path')

    # # Filtering rows where the 'sql' column starts with 'select from'
    # df = df[df[output_column].str.lower().startswith('select from')]
    
    df_main = df.copy()
    chunk_size = 100
    num_chunks = len(df) // chunk_size + 1
    logger.info(f"Processing {len(df)} rows in {num_chunks} chunks.")
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df))
        logger.info(f"Processing chunk {start_index} to {end_index}.")
        chunk_df = df_main[start_index:end_index]    

        output_file = f"{output_path}_{start_index}_{end_index}.parquet" if output_path else None
        if output_file is None or not os.path.exists(output_file):
    
            chunk_df.drop('error', axis=1, errors='ignore', inplace=True)
            chunk_df.drop(apiResultCount_column, axis=1, errors='ignore', inplace=True)
            chunk_df = chunk_df.parallel_apply(
                screening_function, axis=1, 
                input_column=output_column, output_column=screening_column)

            chunk_df = flatten(chunk_df, screening_column)
            # chunk_df = chunk_df[chunk_df.error.isna()] if 'error' in chunk_df.columns else chunk_df

            if output_file:
                try:
                    chunk_df.to_parquet(output_file)
                except Exception as e:
                    logger.error(
                        f"Error saving the embeddings to {output_file}: {repr(e)}")
        else:
            logger.info(
                f"Skipping chunk {start_index} to {end_index} as it already exists.")
        # logger.debug(f'checking for error in the flatten dataframe with {len(chunk_df)} rows.')
        # chunk_df = chunk_df[chunk_df.error == 'nan'] if 'error' in chunk_df.columns else chunk_df
        # if not is_empty_result_acceptable and apiResultCount_column in chunk_df.columns:
        #     rel = duckdb.sql(
        #         f"select * from chunk_df where {apiResultCount_column} != '0.0'")
        #     chunk_df = rel.to_df()            
            
        # if output_path:
        #     output_file = f"{output_path}_screened_{start_index}_{end_index}.parquet"
        #     if not os.path.exists(output_file):
        #         logger.info(f"Saving the screened dataframe to {output_path} with {len(chunk_df)} rows.")
        #         chunk_df.to_parquet(output_file, index=False)    
    

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
    table_fields_mapping_file = config.screening.get(
        'table_fields_mapping_file')

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
                                           axis=1).tolist(), ignore_index=True)

    # screening df through API call

    df = api_screening(df=screened_df, screening_column=screening_column,
                       apiResultCount_column=apiResultCount_column,
                       is_empty_result_acceptable=is_empty_result_acceptable)

    return df

def merge_all_screened_files(pattern: str = None, 
                             output_path: str = None):
    pattern = pattern or config.screening.get('merge_pattern')
    output_path = output_path or config.screening.get('merge_output_path')
    df = merge_parquet_files(pattern)
    df.to_parquet(output_path)


if __name__ == '__main__':
    merge_all_screened_files()
    