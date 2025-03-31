import os
from pandas import DataFrame, concat
from medinote import initialize, read_dataframe, chunk_process
from medinote.embedding.vector_search import search_by_natural_language
from medinote.inference.inference_prompt_generator import row_infer

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

def matching_criteria(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(matching_criteria.__name__)
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
    if df is None:
        df = DataFrame({ "query": [config.get("query")]})
    df = chunk_process(
        df=df,
        function=set_crteria,
        config=config,
        chunk_size=20,
    )
    print(df.shape)
    
def set_crteria(row: dict, config: dict):
    query = row.get("query")
    df, _ = search_by_natural_language(
        query=query,
        config=config,
    )
    if config.get("second_critera"):
        query_template = config.get("second_critera").get("query_template")
        second_df_list = []
        for _, row_x in df.iterrows():
            row_x['query'] = query_template.format(**row_x)
            
            second_df_list.append(set_crteria(row_x, config.get("second_critera")))
        df = concat(second_df_list, ignore_index=True)
    for key, value in row.items():
        include_parent_keys = config.get("include_parent_keys") or []
        if key in include_parent_keys:
            df[key] = value
    return df        

if __name__ == "__main__":
    matching_criteria()