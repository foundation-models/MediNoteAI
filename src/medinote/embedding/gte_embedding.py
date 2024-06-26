import os
from PyPDF2 import PdfReader
from llama_index.core.node_parser import SentenceSplitter
from pandas import DataFrame
import yaml
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
    df['instruct'] = config.get('instruct', 'retrieve relevant passages that answer the query')
    df.loc[df['is_query'], 'embedding_input'] = 'Instruct: ' + df.loc[df['is_query'], 'instruct'] + '\nQuery: ' + df.loc[df['is_query'], 'text']
    df.loc[~df['is_query'], 'embedding_input'] = df.loc[~df['is_query'], 'text']
    df = chunk_process(
        df=df,
        function=row_infer,
        config=config,
        chunk_size=20,
    )
    return df

if __name__ == "__main__":
    gte_embedding()