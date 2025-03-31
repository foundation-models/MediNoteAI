import os
import json
import subprocess
import psycopg2
import pandas as pd
from pandas import DataFrame
from medinote import initialize, read_dataframe
from medinote.embedding.vector_search import execute_query_via_config
import requests
import time

url_yolo_endpoint = "http://48.216.174.39:8000/process_images/"



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
        drop_task_queue_table(config=config)
        create_task_queue_table(config=config)
        if df is not None:
            for _, task in df.iterrows():
                add_new_task(task=task, config=config)

        
    # unique_attribute_index(config=config, unique_attribute="file_name")
                
    # if df is not None:
    #     for _, task in df.iterrows():
    #         add_task_if_not_exist(task=task, config=config, unique_attribute="file_name")
    
            
        #df = push_task(config=config, task_id=1, task={"name": "task1", "status": "new"})      
    return df

def drop_task_queue_table(config: dict):
    execute_query_via_config(config=config, query_key="drop_task_queue_table")

def create_task_queue_table(config: dict): 
    execute_query_via_config(config=config, query_key="create_task_queue_table")

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



def update_task_status_and_json(config: dict, x_status: str, y_key: str, y_value: str, s_status: str, j_key: str, j_value: bool):
    config = main_config.get(sql_based_pipeline.__name__)
    params = {
        "x_status": x_status,
        "y_key": y_key,
        "y_value": y_value,
        "s_status": s_status,
        "j_key": j_key,
        "j_value": j_value
    }

    execute_query_via_config(config=config, query_key="update_task_status_and_json", params=params)

#########################################################################################################################

def check_and_display_new_files():
    config = main_config.get(sql_based_pipeline.__name__)
    folder_path = config.get('folder_path')
    
    # Get existing file names from the database
    df = execute_query_via_config(config=config, query_key="get_existing_file_names")
    current_files = set(file for file in os.listdir(folder_path) if file.endswith('.jpg'))

    existing_files = set(df.iloc[:, 0].dropna())

    new_files = current_files - existing_files

    new_files_data = [{
        "file_name": file,
        "file_path": os.path.join(folder_path, file)
    } for file in new_files]
    
    df_new_files = pd.DataFrame(new_files_data, columns=["file_name", "file_path"])
    if df_new_files is not None:
            for _, task in df_new_files.iterrows():
                add_new_task(task=task, config=config)

####################################################################################################################3

def process_and_update_tasks_DVC(config: dict):
    config = main_config.get(sql_based_pipeline.__name__)
    df = execute_query_via_config(config=config, query_key="get_inferred_jpg_file_paths")

    
    for _, row in df.iterrows():
        file_path = row[1]
        task_id = row[0]
        if file_path is None:
            print(f"File path is missing for task_id {task_id}")
            continue

        # dvc.api.add(file_path) 
        print(f"Added {file_path} to DVC.")
        

        params = {
            "s_status": "dvc",
            "j_key": "dvc",
            "j_value": "true",
            "id": task_id
        }

        execute_query_via_config(config=config, query_key="update_task_with_dvc", params=params)


        # execute_query_via_config(config=config, query_key="update_task_with_dvc", params={
        #     "pending": "pending",
        #     "dvc": "dvc",
        #     "task_id": task_id
        # })
        print(f"Updated task_id {task_id} with DVC status.")



################################################################################################################# yolo 
def yolo_():
    config = main_config.get(sql_based_pipeline.__name__)
    df = execute_query_via_config(config=config, query_key="get_read_jpg_file_paths")
    for _, row in df.iterrows():
        file_path = row[1]
        task_id = row[0]
        if file_path is None:
            print(f"File path is missing for task_id {task_id}")
            continue
        try:
            file_to_send = {"files": (os.path.basename(file_path), open(file_path, 'rb'))}
            response = requests.post(url_yolo_endpoint, files=file_to_send)
            response_json =response.json()
            params = {
                "s_status": "inferred",
                "j_key": "prediction",
                "j_value": response_json[file_path.split("/")[-1]],
                "id": task_id
            }

            execute_query_via_config(config=config, query_key="update_task_with_yolo", params=params)
        except:
            print("error yolo part " + file_path)
###########################################################################################################################
def find_dvc_tasks_one_day_old(config: dict):
    config = main_config.get(sql_based_pipeline.__name__)
    df = execute_query_via_config(config=config, query_key="find_dvc_tasks_one_day_old")
    return df
##########################################################################################################################






if __name__ == "__main__":
    sql_based_pipeline()

    ###############################################
    # x_status = "pending"
    # y_key = "file_name"  
    # y_value = "fastpi_sqlserver.py" 
    # ###############################################
    # s_status = "completed"
    # j_key = "dvc"  
    # j_value = True 

    # update_task_status_and_json(config=main_config, x_status=x_status, y_key=y_key, y_value=y_value, s_status=s_status, j_key=j_key, j_value=json.dumps(j_value))
   


    ##############################################################################################

    ##############################################################################################
    check_and_display_new_files()
    ##############################################################################################
    yolo_()
    ##############################################################################################    
    process_and_update_tasks_DVC(config=main_config)
    ##############################################################################################
    find_dvc_tasks_one_day_old(config=main_config)
    ##############################################################################################
