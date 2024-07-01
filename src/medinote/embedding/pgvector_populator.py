import os
from pandas import DataFrame, DataFramepgvector_populator
from medinote import chunk_process, read_dataframe, write_dataframe, initialize
from medinote.embedding.vector_search import create_or_update_pgvector_table

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=f"{os.path.dirname(__file__)}/../..",
)

def pgvector_populator(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(pgvector_populator.__name__)
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
    df = chunk_process(
        df=df,
        function=create_or_update_pgvector_table,
        config=config,
        chunk_size=10,
    )
    return df

if __name__ == "__main__":
    pgvector_populator()