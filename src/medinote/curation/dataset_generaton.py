
import os
from pandas import DataFrame
import yaml
from medinote import read_dataframe, write_dataframe

try:
    with open(
        os.environ.get("CONFIG_YAML"), "r"
    ) as file:
        conf = yaml.safe_load(file)
        dataset_generation_config = conf.get("datasets_generation")
        if not dataset_generation_config:
            raise Exception("Analysis config not found.")

    def generate_datasets(
        df: DataFrame = None,
        config: dict = None,
    ):

        config = config = dataset_generation_config
        # Perform analysis
        if df is None:
            # Perform analysis on the data
            input_path = config.get("input_path")
            if input_path:
                df = read_dataframe(input_path)
            else:
                raise ValueError("Input path not provided.")
        # Step 1: Sort the DataFrame based on the 'date' column
        sort_column = config.get("sort_column")
        df_sorted = df.sort_values(by=sort_column)

        # Step 2: Determine the sizes for training, validation, and testing sets
        total_samples = len(df_sorted)
        train_size = int(0.7 * total_samples)  # 70% for training
        val_size = int(0.15 * total_samples)   # 15% for validation
        test_size = total_samples - train_size - val_size  # Remaining for testing

        # Step 3: Split the sorted DataFrame into training, validation, and testing sets
        train_df = df_sorted[:train_size]
        val_df = df_sorted[train_size:train_size + val_size]
        test_df = df_sorted[train_size + val_size:]

        # Optionally, you can reset the index for each subset if needed
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        
        output_prefix = config.get("output_prefix")
        if not output_prefix:
            raise ValueError("Output prefix not provided.")
        write_dataframe(train_df, f"{output_prefix}_train.parquet")
        write_dataframe(val_df, f"{output_prefix}_val.parquet")
        write_dataframe(test_df, f"{output_prefix}_test.parquet")
    
except Exception as e:
    print(e)
    raise e    

if __name__ == "__main__":
    generate_datasets()