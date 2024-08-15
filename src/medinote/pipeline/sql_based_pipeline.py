import os
from medinote import chunk_process, read_dataframe, initialize
from pandas import DataFrame
import json

from medinote.embedding.vector_search import construct_insert_command, create_pgvector_table, execute_query, execute_query_via_config, get_embedding

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or f"{os.path.dirname(__file__)}/..",
)

def sql_based_pipeline(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(sql_based_pipeline.__name__)
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
        create_task_queue_table(config=config)
                
    if df is not None:
        for _, task in df.itertasks():
            add_new_task(task=task, config=config)
        return
        
    df = push_task(config=config, task_id=1, task={"name": "task1", "status": "new"})

        
        
    return df


def create_task_queue_table(config: dict): 
    execute_query_via_config(config=config, query_key="create_task_queue_table")

def add_new_task(task: dict, config: dict):
    execute_query_via_config(config=config, query_key="add_new_task", params = { "task": json.dumps(dict(task)) })

def pull_task(config: dict):
    df = execute_query_via_config(config=config, query_key="pull_task")
    return df


def push_task(error_message: str, task_id: int,  config: dict):
    df = execute_query_via_config(config=config, query_key="push_task", params = { "error_message": error_message, "task_id": task_id })
    return df

def log_failure(task_id: int, config: dict):
    execute_query_via_config(config=config, query_key="log_failure", params = { "task_id": task_id })


if __name__ == "__main__":
    sql_based_pipeline()