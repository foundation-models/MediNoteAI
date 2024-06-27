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
    query_condition = config.get('query_condition', 'is_query == True')
    df['instruct'] = config.get('instruct', 'retrieve relevant passages that answer the query')
    df1 = df.query(query_condition).drop_duplicates()
    df1['embedding_input'] = 'Instruct: ' + df1['instruct'] + '\nQuery: ' + df1['text']
    df2 = df.query(f'not ({query_condition})').drop_duplicates()
    df2['embedding_input'] = df2['text']
    df = concat([df1, df2], ignore_index=True)
    # df = df[:2]
    df = chunk_process(
        df=df,
        function=row_infer,
        config=config,
        chunk_size=20,
    )
    return df

if __name__ == "__main__":
    gte_embedding()