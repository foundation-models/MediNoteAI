
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
db = 5
redis_client = redis.Redis(host=redis_host, port=6379, db=db)
app = Celery('tasks', broker=f'redis://{redis_host}:6379/{db}',
             backend=f'redis://{redis_host}:6379/{db}')
topic_prefix = os.environ.get('TOPIC_PREFIX', 'test4')

def get_string_before_last_dot(text):
    if '.' in text:
        return text.rsplit('.', 1)[0]
    else:
        return None
    
# full_function_name = os.environ['FULL_FUNCTION_NAME']

# # Dynamically import the module
# module_name = get_string_before_last_dot(full_function_name)
# logger.info(f"Module name: {module_name}")
# if module_name:
#     module = importlib.import_module(module_name)
#     func_name = full_function_name.split('.')[-1]
#     logger.info(f"Function name: {func_name}")
#     func = getattr(module, func_name, None)
# else:
#     logger.info(f"Function name: {full_function_name}")
#     func = globals()[full_function_name]
from apps.utils.dealcloud_util import get_result_from_sql
func = get_result_from_sql

@app.task(trail=True)
def process_data(data, unique_id: str):
    print(f"receive {unique_id}")
    if unique_id.startswith(topic_prefix):
        print(f"processing ...{data}...{unique_id}..")
        result = get_result_from_sql(data)
        redis_client.set(unique_id, result)
        print("done")
        return True
    else:
        logger.info(f"Ignoring message with topic {data}")
        return False


@task_success.connect # this cause problem
def task_success_handler(sender=None, result=None, **kwargs):
    # Delete task result from Redis
    task_id = sender.request.id
    redis_client.delete(f'celery-task-meta-{task_id}')

    
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
    task_results = []
    for index, row in df.iterrows():
        id = f"{topic_prefix}_{now}_{row[id_column] if id_column else index}"
        print(f"submitting task for {id} {row[text_column]}")
        result = process_data.delay(row[text_column], id)
        task_results.append(result)

    print("Start Waiting ...")
    # Wait for all tasks to complete

    # Wait for all tasks to complete
    for result in task_results:
        result.wait()  # This will block until the specific task is done
        


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

    print("Done Waiting ...")

    print(df.shape)

    # Retrieve results
    for index, row in df.iterrows():
        id = f"{topic_prefix}_{now}_{row[id_column] if id_column else index}"
        print(id)
        result_in_bytes = redis_client.get(id)
        print(f"result_in_bytes: {result_in_bytes}")
        if result_in_bytes:
            result = result_in_bytes.decode('utf-8')
            print(f"result: {result}")
            if result.startswith('{') and result.endswith('}'):
                df.at[index, error_column] = result
            else:
                df.at[index, result_column] = result
        else:
            print("ZZZZZZZZZZZZZZZZZZ")
    print(df.head())
    if save_result_file:
        df.to_parquet(save_result_file)
    if do_delete_redis_entries:
        for index, row in df.iterrows():
            id = f"{topic_prefix}_{now}_{row[id_column] if id_column else index}"
            redis_client.delete(id)

    return df
