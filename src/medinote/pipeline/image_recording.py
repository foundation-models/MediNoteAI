import os
from medinote import chunk_process, read_dataframe, initialize
from pandas import DataFrame
import json
import psycopg2

from medinote.embedding.vector_search import construct_insert_command, create_pgvector_table, execute_query, execute_query_via_config, get_embedding
from medinote.inference.inference_prompt_generator import row_infer

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or f"{os.path.dirname(__file__)}/..",
)

def image_recording(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(image_recording.__name__)
    config["logger"] = logger
    df = (
        df
        if df is not None
        else (
            read_dataframe(config.get("input_path"))
            if config.get("input_path")
            else None
        )
    )

    if config.get("recreate"):
        drop_pipeline_database_table(config=config)
        create_pipeline_database_table(config=config)
    if df is not None:
                for _, task in df.iterrows():
                    add_new_task(task=task, config=config)
        
    # unique_attribute_index(config=config, unique_attribute="file_name")
                
    # if df is not None:
    #     for _, task in df.iterrows():
    #         add_task_if_not_exist(task=task, config=config, unique_attribute="file_name")


def drop_pipeline_database_table(config: dict):
    execute_query_via_config(config=config, query_key="drop_pipeline_database_table")

def create_pipeline_database_table(config: dict): 
    execute_query_via_config(config=config, query_key="create_pipeline_database_table")

def add_new_task(task: dict, config: dict):
    execute_query_via_config(config=config, query_key="add_new_task", params = { "task": json.dumps(dict(task)) })


def unique_attribute_index(config: dict, unique_attribute: str):
    execute_query_via_config(config=config, query_key="unique_attribute_index", params = { "unique_attribute": unique_attribute })


def add_task_if_not_exist(task: dict, config: dict, unique_attribute: str):
    execute_query_via_config(config=config, query_key="add_task_if_not_exist", params = { "task": json.dumps(dict(task)), "unique_attribute": unique_attribute })

def pull_task(config: dict):
    df = execute_query_via_config(config=config, query_key="pull_task")
    return df


def push_task(error_message: str, task_id: int,  config: dict):
    df = execute_query_via_config(config=config, query_key="push_task", params = { "error_message": error_message, "task_id": task_id })
    return df

def log_failure(task_id: int, config: dict):
    execute_query_via_config(config=config, query_key="log_failure", params = { "task_id": task_id })


def update_task(task_id: int, updated_task_data: dict, config: dict):
    params = {
        "task_id": task_id,
        "name_path": json.dumps(updated_task_data)
    }
    execute_query_via_config(query_key="update_task_query", config=config, params=params)


def update_task_table(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(image_recording.__name__)
    config["logger"] = logger
    updated_task_data = {"key": "value"} 
    task_id = 1  
    
    update_task(task_id=task_id, updated_task_data=updated_task_data, config=config)


def fetch_sorted_tasks(config: dict):
    config = config or main_config.get(image_recording.__name__)
    config["logger"] = logger
    execute_query_via_config(query_key="fetch_sorted_tasks_query", config=config)



if __name__ == "__main__":
    #config = main_config.get(image_recording.__name__) 
    image_recording()
    #update_task_table()
    #fetch_sorted_tasks(config=config)

