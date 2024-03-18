import os
from pandas import DataFrame, read_parquet
from medinote import initialize, dynamic_load_function_from_env_varaibale_or_config


config, logger = initialize()


def sample_dataframes(df: DataFrame = None,
                        input_column: str = None,
                        output_column: str = None,
                        output_path: str = None,
                        sample_size: int = 1000
                        ):
        input_column = input_column or config.curate.get('input_column')
        output_column = output_column or config.curate.get('output_column')
        output_path = output_path or config.curate.get('output_path')
        sample_size = sample_size or config.curate.get('sample_size')
        if df is None:
            df = read_parquet(config.curate.get('input_path'))
        df = df.sample(n=sample_size)
        df[output_column] = df[input_column]
        if output_path:
            df.to_parquet(output_path)
        return df

# def fetch_and_save_data():
#     # Read environment variables
#     source_path = os.environ['SOURCE_DATAFRAME_PARQUET_PATH']
#     output_path = os.environ['OUTPUT_DATAFRAME_PARQUET_PATH']
#     text_column = os.environ.get('TEXT_COLUMN', 'text')
#     id_column = os.environ.get('ID_COLUMN')
#     result_column = os.environ.get('output_column', 'result')
#     start_index = os.environ.get('START_INDEX')
#     df_length = os.environ.get('DF_LENGTH')

#     # Read the DataFrame from Parquet file
#     df = pd.read_parquet(source_path)
    
#     if start_index is not None:
#         df = df[int(start_index):]
#     else:
#         start_index = 0
        
#     if df_length is not None:
#         df = df[:int(df_length)]
        

#     ids = asyc_inference_via_celery(df=df,
#                                    text_column=text_column,
#                                    result_column=result_column,
#                                    id_column=id_column,
#                                 #    save_result_file=f"{output_path}_{start_index}_{df_length}.parquet",
#                                       save_result_file='/tmp/ids.txt',
#                                    do_delete_redis_entries=True,
#                                    )
#     sleep(5)
#     ids = collect_celery_task_results(df=df,
#                                    text_column=text_column,
#                                    result_column=result_column,
#                                    id_column=id_column,
#                                 #    save_result_file=f"{output_path}_{start_index}_{df_length}.parquet",
#                                       save_result_file='/tmp/ids.txt',
#                                    do_delete_redis_entries=True,
#                                    )
    # return ids


if __name__ == "__main__":
    sample_dataframes()
