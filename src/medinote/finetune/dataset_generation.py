import json
import os
from pandas import DataFrame, Series, read_parquet
import yaml
from medinote import chunk_process, initialize

_, logger = initialize()


def row_prompt_gen(row: Series, config: dict = None):
    prompt_column = config.get("prompt_column")
    prompt_template = config.get("prompt_template")
    if prompt_column not in row:
        row["text"] = prompt_template.format(**row)
    else:
        row["text"] = row[prompt_column]
    return row
    
try:
    with open(
        os.environ.get("CONFIG_YAML"), "r"
    ) as file:
        finetune_conf = yaml.safe_load(file).get('finetune')
        if not finetune_conf:
            raise Exception("SQL config not found")  
  

    def generate_jsonl_dataset(df: DataFrame = None):
        df = chunk_process(df=df, function=row_prompt_gen, 
                config=finetune_conf, 
                ) 
        output_prefix = finetune_conf.get("output_prefix")
        dataset_path = f"{output_prefix}.jsonl"
        logger.debug(f"Generating dataset {dataset_path} for {df.shape[0]} rows")
        df[["text"]].to_json(dataset_path, orient="records", lines=True)

except Exception as e:
    logger.warning(f"DataCortex API call is not available {repr(e)}")
    
    
    
if __name__ == "__main__":
    generate_jsonl_dataset()
