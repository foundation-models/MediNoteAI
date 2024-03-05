import os
from time import sleep
import pandas as pd
import logging
from medinote.curation.celery_tasks import asyc_inference_via_celery, collect_celery_task_results



# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def fetch_and_save_data():
    # Read environment variables
    source_path = os.environ['SOURCE_DATAFRAME_PARQUET_PATH']
    output_path = os.environ['OUTPUT_DATAFRAME_PARQUET_PATH']
    text_column = os.environ.get('TEXT_COLUMN', 'text')
    id_column = os.environ.get('ID_COLUMN')
    result_column = os.environ.get('output_column', 'result')
    start_index = os.environ.get('START_INDEX')
    df_length = os.environ.get('DF_LENGTH')

    # Read the DataFrame from Parquet file
    df = pd.read_parquet(source_path)
    
    if start_index is not None:
        df = df[int(start_index):]
    else:
        start_index = 0
        
    if df_length is not None:
        df = df[:int(df_length)]
        
    print(df.head())

    ids = asyc_inference_via_celery(df=df,
                                   text_column=text_column,
                                   result_column=result_column,
                                   id_column=id_column,
                                #    save_result_file=f"{output_path}_{start_index}_{df_length}.parquet",
                                      save_result_file='/tmp/ids.txt',
                                   do_delete_redis_entries=True,
                                   )
    sleep(5)
    ids = collect_celery_task_results(df=df,
                                   text_column=text_column,
                                   result_column=result_column,
                                   id_column=id_column,
                                #    save_result_file=f"{output_path}_{start_index}_{df_length}.parquet",
                                      save_result_file='/tmp/ids.txt',
                                   do_delete_redis_entries=True,
                                   )
    return ids


if __name__ == "__main__":
    fetch_and_save_data()
