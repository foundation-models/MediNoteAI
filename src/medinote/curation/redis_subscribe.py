import argparse
import json
import multiprocessing
import os
import threading
from pandas import DataFrame, read_parquet
import redis
import time
from apps.utils.dealcloud_util import get_result_from_sql
from medinote.cached import write_dataframe

# Connect to Redis server
redis_client = redis.Redis(host='redis', port=6379, db=2)


def search(prefix: str):
    # Fetch all keys starting with the prefix
    keys = redis_client.keys(prefix)

    # Sort the keys in descending order
    sorted_keys = sorted(keys, reverse=True)

    # Print the sorted keys
    for key in sorted_keys:
        print(key.decode('utf-8'))

def create_pipeline():
    # Create a pipeline
    pipeline = redis_client.pipeline()
    
    redis_client.set('bing', 'baz')

    # Add commands to the pipeline
    pipeline.set('key1', 'value1')
    pipeline.get('bing')
    
    pipeline.execute()


channel = "test_channel5"


def main():
    df = read_parquet('/mnt/aidrive/datasets/sql_gen/SQL_QA_output_formatted.parquet')
    # Create a thread for the subscribe_to_channel function
    # subscribe_thread = threading.Thread(target=subscribe_to_channel, args=(df,))
    subscribe_process = multiprocessing.Process(target=subscribe_to_channel, args=(df,))

    subscribe_process.start()

    time.sleep(5)
    # Continue with the rest of the program
    publish_message(df)

    
def publish_message(df, id_column, text_column, result_column='result'):

    if id_column is not None and id_column not in df.columns:
        raise ValueError(f"ID column {id_column} not found in DataFrame")
    elif id_column is None:
        df = df.reset_index()
        id_column = 'index'
    else:
        df = df.reset_index().rename(columns={'index': id_column})
         
    count = 0
    
    for _, row in df.iterrows():
        # Publish the message to the channel
        try:
            message = f'{{ \"{id_column}\": \"{row[id_column]}\", \"{text_column}\": \"{row[text_column]}\" }}'
            # Make the message persistent by storing it in a list
            # redis_client.rpush(f"persistent_messages_{channel}", message)

            # Publish the message to the channel
            redis_client.publish(channel, message)
            count += 1
            print(count)
            # print(f"Message {message} published and persisted to channel '{channel}'")

        except Exception as e:
            print(f"Error publishing message: {e}")
            
            
    
    
def subscribe_to_channel(df, output_path, id_column, text_column, result_column='result'):

    if id_column is not None and id_column not in df.columns:
        raise ValueError(f"ID column {id_column} not found in DataFrame")
    elif id_column is None:
        df = df.reset_index()
        id_column = 'index'
    else:
        df = df.reset_index().rename(columns={'index': id_column})


    # Create a pubsub object
    pubsub = redis_client.pubsub()
    
    # Subscribe to the channel
    pubsub.subscribe(channel)
            
    try:

        channels = redis_client.pubsub_channels()
        print("Active channels:", channels)

        # Subscribe to the channel
        pubsub = redis_client.pubsub()
        pubsub.subscribe(channel)
        print(f"Subscribed to channel '{channel}'")
        
        count = 1

        while True:
            # Check for new message
            message = pubsub.get_message()
            print(message)
            if message and message['type'] == 'message':
                try:

                    message_data = message['data'].decode('utf-8')
                    message_data_dict = json.loads(message_data)
                    index = int(message_data_dict.get(id_column))
                    sqlQuery = message_data_dict.get(text_column)


                    # Get result and SQL query
                    result, sqlquery = get_result_from_sql(sqlQuery=sqlQuery)

                # Update the DataFrame
                    df.at[index, text_column] = sqlquery
                    df.at[index, result_column] = str(result)
                    count += 1
                    print(count)
                    if count == df.shape[1]:
                        break
    
                    # Read and remove the top message from the persistent list
                    # removed_message = redis_client.lpop(f"persistent_messages_{channel}")
                except Exception as e:
                    print(f"ErrorXXXX: {e}")
                                
            time.sleep(1)  # Wait for 1 second before checking for new messages
        write_dataframe(df=df,output_path=output_path)
        
        
    except Exception as e:
        print(f"Error: {e}")

                   
def main():
    source_path = os.environ['SOURCE_DATAFRAME_PARQUET_PATH']
    output_path = os.environ['OUTPUT_DATAFRAME_PARQUET_PATH']
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
    
    subscribe_to_channel(df=df, output_path=output_path, text_column=text_column, id_column=id_column, result_column=result_column)
    
if __name__ == "__main__":
    main()