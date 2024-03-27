

import duckdb
from pandas import DataFrame, read_parquet
from medinote import dynamic_load_function_from_env_varaibale_or_config, initialize, merge_parquet_files


config, logger = initialize()


""" 
develoging based on this plan: 
https://chat.openai.com/share/b5cc5846-141a-4b57-8560-8065236552d8
"""

def search_df(obj_name: str):
    """_summary_
    
    Search a DataFrame for a specified object name column.
    """
    pattern = f"%{obj_name[1:-1]}%"
    input_path = config.dataframe_search.get("input_path")
    column_names = config.dataframe_search.get("column_names") or "*"
    rel = duckdb.query(f"select {column_names} from read_parquet('{input_path}') where text like '{pattern}'") 
    return rel.df()
    
if __name__ == "__main__":
    search_df("Company")
