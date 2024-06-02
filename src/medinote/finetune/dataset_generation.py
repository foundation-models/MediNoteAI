import os
from pandas import DataFrame, Series, read_parquet
import yaml
from medinote import chunk_process, initialize, read_dataframe, write_dataframe

_, logger = initialize()

with open(
    os.environ.get("CONFIG_YAML")
    or f"{os.path.dirname(__file__)}/../config/config.yaml",
    "r",
) as file:
    main_config = yaml.safe_load(file)


def generate_jsonl_dataset(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(generate_jsonl_dataset.__name__)
    df = (
        df
        if df is not None
        else (
            read_dataframe(config.get("input_path"))
            if config.get("input_path")
            else None
        )
    )
    # df = df[:4]
    df = chunk_process(
        df=df,
        function=row_prompt_gen,
        config=config,
    )
    output_prefix = config.get("output_prefix")
    jsonl_path = f"{output_prefix}.jsonl".replace("chunks", "merged")
    output_path = f"{output_prefix}.parquet".replace("chunks", "merged")

    logger.info(f"Generating dataset {jsonl_path} for {df.shape[0]} rows")
    df[["text"]].to_json(jsonl_path, orient="records", lines=True)
    write_dataframe(df, output_path)
    return df


def row_prompt_gen(row: Series, config: dict = None):
    prompt_column = config.get("prompt_column")
    prompt_template = config.get("prompt_template")
    if prompt_column is None and prompt_column not in row:
        row["text"] = prompt_template.format(**row)
    else:
        row["text"] = row[prompt_column]
    return row

if __name__ == "__main__":
    generate_jsonl_dataset()
