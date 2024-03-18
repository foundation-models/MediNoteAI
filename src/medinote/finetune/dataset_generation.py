import json
from pandas import DataFrame, Series, read_parquet
from medinote import initialize

config, logger = initialize()

def generate_jsonl_dataset(df: DataFrame = None,
                     prompt_column: str = None,
                     ):
    df = df or read_parquet(config.finetune['dataset_input_path'])
    prompt_column = prompt_column or config.finetune['prompt_column']
    dataset_path = config.finetune['dataset_output_path']
    logger.debug(f"Generating dataset {dataset_path} for {df.shape[0]} rows")
    df["text"]  = df[prompt_column]   
    df[["text"]].to_json(dataset_path, orient='records', lines=True)

if __name__ == "__main__":
    generate_jsonl_dataset()
