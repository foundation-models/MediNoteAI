import os
from medinote import chunk_process, read_dataframe, initialize
from pandas import DataFrame

from medinote.embedding.vector_search import construct_insert_command, create_pgvector_table, execute_query, get_embedding

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or f"{os.path.dirname(__file__)}/..",
)

def sql_based_queue(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(sql_based_queue.__name__)
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
        creare_table(config=config)
    df = chunk_process(
        df=df,
        function=create_or_update_pgvector_table,
        config=config,
        chunk_size=30,
    )
    return df


def creare_table(config: dict):
    create_table_statement = config.get("create_table_statement")
    execute_query(create_table_statement)
    


if __name__ == "__main__":
    sql_based_queue()