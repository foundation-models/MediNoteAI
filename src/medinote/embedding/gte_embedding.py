import json
import os
from pandas import DataFrame, concat
from medinote.inference.inference_prompt_generator import row_infer
from medinote import initialize, read_dataframe, chunk_process

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or f"{os.path.dirname(__file__)}/..",
)

def gte_embedding(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(gte_embedding.__name__)
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
    to_embed_column = config.get('to_embed_column', 'text')
    if to_embed_column not in df.columns:
        raise ValueError(f'Column "{to_embed_column}" not found in the dataframe')
    
    query_condition = config.get('query_condition', 'is_query == True')
    if 'is_query' in df.columns:
        df['instruct'] = config.get('instruct', 'retrieve relevant passages that answer the query')
        df1 = df.query(query_condition).drop_duplicates()
        df1['embedding_input'] = 'Instruct: ' + df1['instruct'] + '\nQuery: ' + df1[to_embed_column]
        df2 = df.query(f'not ({query_condition})').drop_duplicates()
        df2['embedding_input'] = df2[to_embed_column]
        df = concat([df1, df2], ignore_index=True)
    else:
        df['embedding_input'] = df[to_embed_column]
    if 'file_path' in df.columns:
        df['file_name'] = df['file_path'].apply(lambda x: os.path.basename(x))
    if key_values := config.get("key_values"):
        for key, value in key_values.items():
            df[key] = value
    df = chunk_process(
        df=df,
        function=row_embedding,
        config=config,
        chunk_size=20,
    )
    return df

def row_embedding(row: dict, config: dict):
    row = row_infer(row, config)
    if embedding_element:=row.get('embedding'):
        embedding_json = json.loads(embedding_element)
        if data:=embedding_json.get('data'):
            row['embedding'] = data[0].get('embedding')
        else:
            logger.error(f'No data found in the embedding element: {embedding_element}')
            row['embedding'] = None
    else:
        logger.error(f'No embedding element found in the row: {row}')
        row['embedding'] = None
    if not isinstance(row.get('embedding'), list):
        raise ValueError(f'Invalid embedding found in the row: {row}')
    return row
        

if __name__ == "__main__":
    gte_embedding()