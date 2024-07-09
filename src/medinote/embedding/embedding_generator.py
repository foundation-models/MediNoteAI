import os
from pandas import DataFrame, concat
from medinote import initialize, read_dataframe, chunk_process
from medinote.inference.inference_prompt_generator import row_infer

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or f"{os.path.dirname(__file__)}/..",
)

def embedding_generator(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(embedding_generator.__name__)
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
    column2embed = config.get('column2embed', 'text')
    if column2embed not in df.columns:
        raise ValueError(f'Column "{column2embed}" not found in the dataframe')
    
    query_condition = config.get('query_condition', 'is_query == True')
    if 'is_query' in df.columns:
        df['instruct'] = config.get('instruct', 'retrieve relevant passages that answer the query')
        df1 = df.query(query_condition).drop_duplicates()
        df1['embedding_input'] = 'Instruct: ' + df1['instruct'] + '\nQuery: ' + df1[column2embed]
        df2 = df.query(f'not ({query_condition})').drop_duplicates()
        df2['embedding_input'] = df2[column2embed]
        df = concat([df1, df2], ignore_index=True)
    else:
        df['embedding_input'] = df[column2embed]
    if 'file_path' in df.columns:
        df['file_name'] = df['file_path'].apply(lambda x: os.path.basename(x))
    if key_values := config.get("key_values"):
        for key, value in key_values.items():
            df[key] = value
    df = chunk_process(
        df=df,
        function=row_infer,
        config=config,
        chunk_size=20,
    )
    return df
        

if __name__ == "__main__":
    embedding_generator()