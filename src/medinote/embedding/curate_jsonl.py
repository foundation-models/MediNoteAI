import os
from llama_index.core.node_parser import SentenceSplitter
from pandas import DataFrame
from medinote import initialize, read_dataframe, write_dataframe

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or f"{os.path.dirname(__file__)}/../..",
)

def curate_jsonl(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(curate_jsonl.__name__)
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
    rename_columns = config.get("rename_columns", {})
    if rename_columns:
        df = df.rename(columns=rename_columns)
    if output_path := config.get("output_path"):
        write_dataframe(df, output_path)
    

if __name__ == "__main__":
    curate_jsonl()