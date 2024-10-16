import os
from pandas import read_parquet
from medinote.curation.redis_subscribe import publish_message


def main():
    # Read environment variables
    source_path = os.environ['SOURCE_DATAFRAME_PARQUET_PATH']
    text_column = os.environ.get('TEXT_COLUMN', 'text')
    id_column = os.environ.get('ID_COLUMN')
    result_column = os.environ.get('RESULT_COLUMN', 'result')
    start_index = os.environ.get('START_INDEX')
    df_length = os.environ.get('DF_LENGTH')

    # Read the DataFrame from Parquet file
    df = read_parquet(source_path)
    
    if start_index is not None:
        df = df[int(start_index):]
    else:
        start_index = 0
                
    if df_length is not None:
        df = df[:int(df_length)]
        
    # Continue with the rest of the program
    publish_message(df=df, text_column=text_column, id_column=id_column, result_column=result_column)
                   

    
if __name__ == "__main__":
    main()