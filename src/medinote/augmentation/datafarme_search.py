

import duckdb
from pandas import DataFrame, concat, read_parquet
from medinote import dynamic_load_function_from_env_varaibale_or_config, initialize
from medinote.cached import read_dataframe, write_dataframe


config, logger = initialize()

list_obj_names_function = dynamic_load_function_from_env_varaibale_or_config("list_obj_names_function")

""" 
develoging based on this plan: 
https://chat.openai.com/share/b5cc5846-141a-4b57-8560-8065236552d8
"""

def search_df(obj_name: str, df: DataFrame = None):
    """_summary_
    
    Search a DataFrame for a specified object name column.
    """
    pattern = f"%{obj_name[1:-1]}%"
    input_path = config.get("dataframe_search").get("input_path")
    if df is None and input_path:
        df = read_parquet(input_path)
    column_names = config.get("dataframe_search").get("column_names") or "*"
    rel = duckdb.query(f"select {column_names} from df where text like '{pattern}'") 
    # rel = duckdb.query(f"select {column_names} from read_parquet('{input_path}') where text like '{pattern}'") 
    result_df = rel.df()
    result_df['obj_name'] = obj_name
    
    return result_df



def search_df_for_all_objects(persist: bool = True):
    objec_names = list_obj_names_function()
    if not objec_names:
        raise ValueError(f"No object names found.")
    input_path = config.get("dataframe_search").get("input_path")
    if input_path:
        df = read_parquet(input_path)
    dataframes = []
    for obj_name in objec_names:
        try:
            dataframes.append(search_df(obj_name, df=df))
        except Exception as e:
            logger.error(f"Failed to include obj)name {obj_name}: {e}")
            
    df = concat(dataframes)
    
    output_path = config.get("dataframe_search").get('output_path')
    if persist and output_path:
        write_dataframe(output_path)
    
    return df
    
if __name__ == "__main__":
    search_df_for_all_objects()
