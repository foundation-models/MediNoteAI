import json
from celery import Celery
from celery.signals import task_success
from pandas import DataFrame
import redis
import os
import requests
import logging
import datetime

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

redis_host = os.environ.get('REDIS_HOST', 'redis')
db = 1
redis_client = redis.Redis(host=redis_host, port=6379, db=db)
app = Celery('tasks', broker=f'redis://{redis_host}:6379/{db}', backend=f'redis://{redis_host}:6379/{db}')
inference_url = os.environ.get('BASE_CURATION_URL', 'http://llm:8888/worker_generate')  # Default if not set

def fetch_url(text: str, 
              method: str = "post", 
              payload: dict = None, 
              headers: dict = None, 
              token: str = None):
    try:
        headers = headers or {
            "Content-Type": "application/json",
        }
        print(f"Fetching ....{inference_url}....{token}... {method}")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = payload or {
            "echo": False,
            "stop": [
                "<|im_start|>"
            ],
            "prompt": f"<|im_start|>system\nYou are \"Hermes 2\", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>\n<|im_start|>user\nCreate a very short, couple of words note for the following text: \n\n{text}<|im_end|>\n<|im_start|>assistant\n"
        }

        if method.lower() == "post":
            response = requests.post(url=inference_url, headers=headers, json=payload)
        else:
            url = f"{inference_url}/{text}"
            response = requests.get(url=url, headers=headers)     
        print(response.json())
        return response.json()['text'] if 'text' in response.json() else json.dumps(response.json())
    except requests.RequestException as e:
        print(f"Error fetching URL {inference_url}: {e}")
        logger.error(f"Error fetching URL {inference_url}: {e}")
        return json.dumps({"error": f"Error fetching URL {inference_url}: {e}"})
    except Exception as e:
        print(f"Error on the response:\n \n from {inference_url}:\n {e}")
        logger.error(f"Error on the response:\n \n from {inference_url}:\n {e}")
        return json.dumps({"error": f"Error on the response:\n \n from {inference_url}:\n {e}"})
    
topic_prefix = os.environ.get('TOPIC_PREFIX', 'test4')

@app.task
def process_data(data, unique_id):
    print(topic_prefix)
    if unique_id.startswith(topic_prefix):
        print("Processing")
        result = fetch_url(data)
        redis_client.set(unique_id, result)
    else:
        print("Ignoring")
        logger.info(f"Ignoring message with topic {data}")
        
@task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    # Delete task result from Redis
    task_id = sender.request.id
    redis_client.delete(f'celery-task-meta-{task_id}')

def asyc_inference_via_celery(  df: DataFrame, 
                                text_column: str, 
                                id_column: str, 
                                result_column: str, 
                                error_column: str = 'Error',
                                save_result_file: str = None,
                                do_delete_redis_entries = False
                              ):
    
    # Setup Redis client
    now = datetime.datetime.now()
        
    # Enqueue tasks and collect AsyncResult objects
    task_results = []
    for index, row in df.iterrows():
        id = f"{topic_prefix}_{now}_{row[id_column] if id_column else index}"
        result = process_data.delay(row[text_column], id)
        task_results.append(result)
        
    # Wait for all tasks to complete
    for result in task_results:
        result.wait()  # This will block until the specific task is done
        
                
    # Retrieve results
    for index, row in df.iterrows():
        id = f"{topic_prefix}_{now}_{row[id_column] if id_column else index}"
        result_in_bytes = redis_client.get(id)
        if result_in_bytes:
            result = result_in_bytes.decode('utf-8')
            if result.startswith('{') and result.endswith('}'):
                df.at[index, error_column] = result
            else:
                df.at[index, result_column] = result
    if save_result_file:            
        df.to_parquet(save_result_file)
    if do_delete_redis_entries:
        for index, row in df.iterrows():
            id = f"{topic_prefix}_{now}_{row[id_column] if id_column else index}"
            redis_client.delete(id)

    return df
