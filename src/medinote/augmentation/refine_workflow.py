# Generatd with CHatGPT on 2021-08-25 15:00:00 https://chat.openai.com/share/133de26b-e5f5-4af8-a990-4a2b19d02254
from datetime import datetime
from pandas import DataFrame, concat, read_parquet
from medinote import initialize, dynamic_load_function_from_env_varaibale_or_config

config, logger = initialize()


augment_function = dynamic_load_function_from_env_varaibale_or_config(
    'augment_function')


def generate_df(df: DataFrame, error_column: str = 'error'):
    """
    Generates GOOD and BAD df based on a condition in a specific column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    error_column (str): The name of the column to check the condition.

    Returns:
    good_df (pd.DataFrame): Dataset where the specified column is not None.
    bad_df (pd.DataFrame): Dataset where the specified column is None.
    """
    # Check if the specified column exists in the DataFrame
    if error_column not in df.columns:
        raise ValueError(
            f"Column '{error_column}' not found in DataFrame.")

    # Splitting the DataFrame into GOOD and BAD df
    bad_df = df[df[error_column].notna() & (
        df[error_column].apply(lambda x: isinstance(x, str) and x != 'nan'))]
    good_df = df[df[error_column].isna() | (
        df[error_column].astype(str) == 'nan')]

    return good_df, bad_df

# Borrowed from https://docs.llamaindex.ai/en/stable/examples/vector_stores/OpensearchDemo.html


def combine_datasets(good_df, bad_df, size: int = None):
    """
    Randomly picks df with an 80% GOOD and 20% BAD ratio.

    Parameters:
    good_df (pd.DataFrame): The GOOD df.
    bad_df (pd.DataFrame): The BAD df.
    size (int): Total number of df to pick.

    Returns:
    pd.DataFrame: A DataFrame containing the randomly picked df.
    """
    # Calculating the number of GOOD and BAD df to pick

    size = size or config.refine_workflow.get('combined_df_size')

    size = min(size, len(good_df))

    num_good = int(size * 0.8)
    num_bad = size - num_good

   # Randomly sampling GOOD and BAD df
    good_sample = good_df.sample(n=num_good)
    bad_sample = bad_df.sample(n=num_bad)

    # Concatenating the samples into a single DataFrame
    result = concat([good_sample, bad_sample])

    return result


def augment_dataframe(df: DataFrame,
                      template: str = None,
                      inference_response_limit: int = 100,
                      instruction: str = None,
                      output_column: str = None
                      ):
    # Augment df 100 times with GPT call
    template = template or config.augmentation.get('prompt_template')
    inference_response_limit = inference_response_limit or config.augmentation.get(
        'inference_response_limit')
    instruction = instruction or config.augmentation.get('instruction')
    output_column = output_column or config.augmentation.get('output_column')
    inference_url = config.augmentation.get('inference_url')
    payload_template = config.augmentation.get('payload_template')
    output_separator = config.augmentation.get('output_separator')
    table_fields_mapping_file = config.screening.get(
        'table_fields_mapping_file')

    result_df = df.parallel_apply(augment_function, axis=1,
                                  template=template,
                                  inference_response_limit=inference_response_limit,
                                  instruction=instruction,
                                  inference_url=inference_url,
                                  payload_template=payload_template,
                                  output_column=output_column,
                                  output_separator=output_separator,
                                  table_fields_mapping_file=table_fields_mapping_file,
                                  )

    return result_df


def main():
    now = datetime.now().replace(microsecond=0).isoformat().replace(':', '-')

    logger.debug(f"Starting the refine workflow at {now}")
    input_path = config.refine_workflow.get('input_path')

    logger.debug(f"Reading the input parquet file from {input_path}")
    df = read_parquet(input_path)

    # df = df[:1000]
    logger.debug(f"Read {len(df)} rows from the input parquet file")
    bad_df_length = config.refine_workflow.get('bad_df_length')

    logger.debug(f"Generating GOOD and BAD datasets")
    good_df, bad_df = generate_df(df=df)
    logger.debug(f"GOOD dataset has {len(good_df)} rows")
    bad_df = bad_df[:bad_df_length]
    logger.debug(f"BAD dataset has {len(bad_df)} rows")

    # logger.debug(f"Creating vector index using the GOOD dataset")
    # vector_index = create_vector_db_collections(df=good_df)

    count = 0

    logger.debug(f"Starting the loop to refine the dataset")
    while len(bad_df) > 0 and len(good_df) < 15000:
        count += 1
        output_path = config.refine_workflow.get('output_path')
        output_prefix = f"{output_path}_{now}_{count}"

        logger.debug(f"Starting iteration {count}")
        logger.debug(f"Combining GOOD and BAD datasets")
        combined_df = combine_datasets(
            good_df=good_df, bad_df=bad_df)
        # combined_df = combined_df[:10]

        logger.debug(f"Augmenting the combined dataset")
        augmented_df = augment_dataframe(combined_df)
        logger.debug(f"Saving the augmented dataset to a parquet file")
        augmented_df.to_parquet(f"{output_prefix}_augmented.parquet")

        logger.debug(f"Screening the augmented dataset")
        good_results = screeen_dataframes(augmented_df)
        logger.debug(f"Saving the screened results to a parquet file")
        good_results.to_parquet(f"{output_prefix}__screened.parquet")

        logger.debug(f"Adding the good results to the GOOD dataset")
        input_column = config.refine_workflow.get('input_column')

        logger.debug(f"Excluding the good results from the BAD dataset")
        values_to_exclude = good_results[input_column]
        logger.debug(
            f"Excluding {len(values_to_exclude)} rows from the BAD dataset")
        bad_df = bad_df[~bad_df[input_column].isin(values_to_exclude)]
        logger.debug(f"Adding the good results to the GOOD dataset")
        good_df = concat([good_df, good_results])

        output_parquet_file = f"{output_path}_{now}_{count}.parquet"
        logger.debug(
            f"Saving the GOOD dataset to a parquet file at {output_parquet_file}")
        good_df.to_parquet(output_parquet_file)

        # logger.debug(f"Adding the good results to the vector index")
        # add_vector_db_collections(df=good_results, vector_index=vector_index)

    # if is_collection_size_met(future_collection, len(now_collection) * x_percent):
    #     remove_vectors_from_now(now_collection, len(future_collection))

    # if len(good_collection) >= 15000:
    #     refine_sql_model()

    # if len(bad_df) == 0:
    #     # Add bad queries to BAD dataset until it reaches 15K
    #     pass

    # Optionally, rerun the entire process


if __name__ == "__main__":
    main()
