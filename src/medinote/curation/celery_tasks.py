
from celery import Celery
from celery.signals import task_success
from pandas import DataFrame
import redis
import os
import logging
import datetime
import importlib
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

redis_host = os.environ.get('REDIS_HOST', 'redis')
db = 10
redis_client = redis.Redis(host=redis_host, port=6379, db=db)
app = Celery('tasks', broker=f'redis://{redis_host}:6379/{db}',
             backend=f'redis://{redis_host}:6379/{db}')
topic_prefix = os.environ.get('TOPIC_PREFIX', 'test4')


func = get_result_from_sql

@app.task(trail=True)
def process_data(data, unique_id: str):
    print(f"processing ...{data}...{unique_id}..")
    result, sql_query = get_result_from_sql(data)
    redis_client.set(unique_id, str(result))
    print("done")


# @task_success.connect # this cause problem
# def task_success_handler(sender=None, result=None, **kwargs):
#     # Delete task result from Redis
#     task_id = sender.request.id
#     redis_client.delete(f'celery-task-meta-{task_id}')

    
def asyc_inference_via_celery(df: DataFrame,
                              text_column: str,
                              id_column: str,
                              result_column: str,
                              error_column: str = 'Error',
                              save_result_file: str = None,
                              do_delete_redis_entries=False,
                              ):
        
    # Setup Redis client
    now = datetime.datetime.now()

    # Enqueue tasks and collect AsyncResult objects
    ids = []
    for index, row in df.iterrows():
        id = f"{topic_prefix}.{now}:{row[id_column] if id_column else index}"
        print(f"submitting task for {id} {row[text_column]}")
        process_data.delay(row[text_column], id)
        ids.append(id)
        
    print(f"Submitted {len(ids)} tasks")
    print(save_result_file)
    # Write ids to save_result_file
    if save_result_file:
        with open(save_result_file, 'w') as file:
            for index in ids:
                file.write(f"{index}\n")

    # print("Start Waiting ...")
    # # Wait for all tasks to complete

    # # Wait for all tasks to complete
    # for result in task_results:
    #     result.wait()  # This will block until the specific task is done
        


    # # # Define a timeout for each task, in seconds
    # task_timeout = 5 # Adjust as needed


    # # Initialize the start time for timeout tracking
    # start_time = time.time()

#     while True:
#         # Check if the current time exceeds the start time by the timeout duration
#         if time.time() - start_time > task_timeout:
#             print(f"Task {result.id} timed out.")
#             break

#         # Check the status of the task
#         if result.ready():
# #             # Task has finished, retrieve the result
# #             try:
# #                 task_result = result.get(timeout=1)  # Short timeout to fetch the result
# #                 print(f"Task {result.id} completed with result: {task_result}")
# #             except Exception as e:
# #                 print(f"Error getting result for task {result.id}: {e}")
#             break
# #         else:
# #             # The task is not ready yet, you might want to log this or just pass
# #             print(f"Waiting for task {result.id} to complete...")
# #             time.sleep(1)  # Sleep for a short duration before checking again

    # print("Done Waiting ...")
    
def collect_celery_task_results(df: DataFrame,
                              text_column: str,
                              id_column: str,
                              result_column: str,
                              error_column: str = 'Error',
                              save_result_file: str = None,
                              do_delete_redis_entries=False,
                              ):

    print(df.shape)
    
    if save_result_file:
        with open(save_result_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                id = line.strip()
                print(id)    
                result_in_bytes = redis_client.get(id)
                print(f"result_in_bytes: {result_in_bytes}")
                if result_in_bytes:
                    result = result_in_bytes.decode('utf-8')
                    index = id.split(':')[-1]
                    df.at[index, result_column] = result
                    redis_client.delete(id)
                else:
                    print("ZZZZZZZZZZZZZZZZZZ")
    print(df.head())
    if save_result_file:
        df.to_parquet(save_result_file)

            

    return df


