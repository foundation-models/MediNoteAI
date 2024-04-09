import os
from pandas import DataFrame, Series, concat, merge, read_parquet
from medinote import initialize
from medinote import write_dataframe
from medinote.curation.rest_clients import generate_via_rest_client
from pandas import DataFrame, concat, read_parquet
from medinote import initialize
import pyarrow.parquet as pq


_, logger = initialize()


def generate_synthetic_data(row: Series, config: dict = None):
    """_summary_

    Generate synthetic data based on a specified object name.
    """
    try:
        input_column = config.get("curate").get("input_column") or "text"
        input = row[input_column]

        row_dict = {"input": input}

        template = config.get("curate")["brief_narrative_generation_prompt_template"]
        logger.debug(f"Using template: {template}")
        prompt = template.format(**row_dict)

        prompt_column = config.get("curate").get("prompt_column") or "prompt"
        if prompt_column:
            row[prompt_column] = prompt

        template = config.get("curate").get("payload_template")
        payload = template.format(**{"prompt": prompt})

        inference_url = config.get("curate").get("inference_url")
        response = generate_via_rest_client(
            payload=payload, inference_url=inference_url
        )
        output_column = config.get("curate").get("output_column") or "inference"

        row[output_column] = response.replace("\n", " ").strip()

        return row
    except Exception as e:
        logger.error(f"Error generating synthetic data: {repr(e)}")
        return row


def parallel_generate_synthetic_data(df: DataFrame = None, config: dict = None):
    """_summary_

    Generate synthetic data based on a specified object name.
    """
    output_prefix = config.get("curate").get("output_prefix")
    if df is None:
        df = read_parquet(config.get("curate").get("sample_output_path"))

    chunk_size = 10
    num_chunks = len(df) // chunk_size + 1

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df))
        chunk_df = df[start_index:end_index]

        output_file = (
            f"{output_prefix}_{start_index}_{end_index}.parquet"
            if output_prefix
            else None
        )
        if output_file is None or not os.path.exists(output_file):
            try:
                chunk_df = chunk_df.parallel_apply(generate_synthetic_data, axis=1)
            except ValueError as e:
                if "Number of processes must be at least 1" in str(e):
                    logger.error(
                        f"No idea for error: Number of processes must be at least \n ignoring ....."
                    )
            except Exception as e:
                logger.error(f"Error generating synthetic data: {repr(e)}")

            if output_file:
                try:
                    chunk_df.to_parquet(output_file)
                except Exception as e:
                    logger.error(
                        f"Error saving the embeddings to {output_file}: {repr(e)}"
                    )
        else:
            logger.info(
                f"Skipping chunk {start_index} to {end_index} as it already exists."
            )


def read_large_dataframe_columns(
    input_path: str = None, output_path: str = None, config: dict = None
):

    # Adjust the chunk_size according to your memory constraints
    input_path = input_path or config.get("curate").get("input_column")
    output_path = output_path or config.get("curate").get("output_path")

    parquet_file = pq.ParquetFile(input_path)

    dataframes = []
    for i in range(0, parquet_file.num_row_groups):
        df = parquet_file.read_row_group(
            i,
            columns=[
                "client",
                "matter",
                "narrative",
                "billed",
                "approved",
                "deleted",
                "status",
            ],
        ).to_pandas()
        df = df[df["deleted"] == False]
        df = df[df["billed"] == True]
        dataframes.append(df)
    df = concat(dataframes)
    if output_path:
        write_dataframe(df=df, output_path=output_path)
    return df


def sample_large_dataframe(
    input_path: str = None,
    output_path: str = None,
    persist: bool = True,
    config: dict = None,
):

    # Adjust the chunk_size according to your memory constraints
    input_path = input_path or config.get("curate").get("output_path")
    output_path = output_path or config.get("curate").get("sample_output_path")
    sample_size = config.get("curate").get("sample_size")

    parquet_file = pq.ParquetFile(input_path)

    dataframes = []
    for i in range(0, parquet_file.num_row_groups):
        df = parquet_file.read_row_group(
            i,
        ).to_pandas()
        dataframes.append(df)
    df = concat(dataframes)
    df = df.sample(n=sample_size)
    if persist and output_path:
        write_dataframe(df=df, output_path=output_path)
    return df


def sample_dataframes(
    df: DataFrame = None,
    input_column: str = None,
    output_column: str = None,
    output_prefix: str = None,
    sample_size: int = 1000,
    config: dict = None,
):
    if df is None:
        df = read_parquet(config.get("curate").get("input_path"))

    input_column = input_column or config.get("curate").get("input_column")
    output_column = output_column or config.get("curate").get("output_column")
    output_prefix = output_prefix or config.get("curate").get("output_prefix")
    sample_size = sample_size or config.get("curate").get("sample_size")

    df = df.sample(n=sample_size)
    df[output_column] = df[input_column]
    if output_prefix:
        df.to_parquet(output_prefix)
    return df


# def fetch_and_save_data():
#     # Read environment variables
#     source_path = os.environ['SOURCE_DATAFRAME_PARQUET_PATH']
#     output_prefix = os.environ['OUTPUT_DATAFRAME_PARQUET_PATH']
#     text_column = os.environ.get('TEXT_COLUMN', 'text')
#     id_column = os.environ.get('ID_COLUMN')
#     result_column = os.environ.get('output_column', 'result')
#     start_index = os.environ.get('START_INDEX')
#     df_length = os.environ.get('DF_LENGTH')

#     # Read the DataFrame from Parquet file
#     df = pd.read_parquet(source_path)

#     if start_index is not None:
#         df = df[int(start_index):]
#     else:
#         start_index = 0

#     if df_length is not None:
#         df = df[:int(df_length)]


#     ids = asyc_inference_via_celery(df=df,
#                                    text_column=text_column,
#                                    result_column=result_column,
#                                    id_column=id_column,
#                                 #    save_result_file=f"{output_prefix}_{start_index}_{df_length}.parquet",
#                                       save_result_file='/tmp/ids.txt',
#                                    do_delete_redis_entries=True,
#                                    )
#     sleep(5)
#     ids = collect_celery_task_results(df=df,
#                                    text_column=text_column,
#                                    result_column=result_column,
#                                    id_column=id_column,
#                                 #    save_result_file=f"{output_prefix}_{start_index}_{df_length}.parquet",
#                                       save_result_file='/tmp/ids.txt',
#                                    do_delete_redis_entries=True,
#                                    )
# return ids


if __name__ == "__main__":
    parallel_generate_synthetic_data()
