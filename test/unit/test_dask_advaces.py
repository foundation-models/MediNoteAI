from numpy import random
import dask
from medinote import initialize, initialize_worker
from pandas import DataFrame, merge, read_parquet
import random
from medinote.embedding.vector_search import (
    calculate_average_source_distance,
    get_dataset_dict_and_df,
)
from pandas import DataFrame, read_parquet


@dask.delayed
def calculate_average_source_distance(
    exclude_ids: set,
    config: dict = None,
    df: DataFrame = None,
    source_column: str = "source_ref",
    near_column: str = "near_ref",
    distance_column: str = "distance",
    source_distance_column: str = "source_distance",
):
    """
    Calculates the average source distance between documents in a DataFrame.

    Args:
        df (DataFrame, optional): The input DataFrame. If not provided, it will be read from the configured output path.
        source_column (str, optional): The name of the column representing the source reference. Defaults to 'source_ref'.
        near_column (str, optional): The name of the column representing the near reference. Defaults to 'near_ref'.
        distance_column (str, optional): The name of the column representing the distance. Defaults to 'distance'.
        source_distance_column (str, optional): The name of the column to store the calculated average source distance. Defaults to 'source_distance'.
        exclude_ids (list, optional): A list of IDs to exclude from the calculation. Defaults to an empty list.

    Returns:
        DataFrame: The updated DataFrame with the calculated average source distance.

    """
    if df is None:
        output_path = config.get("embedding").get("cross_distance_output_path")
        if output_path:
            df = read_parquet(output_path)
    # df = read_parquet("/home/agent/workspace/rag/dataset/docs_cross_distance.parquet")

    # df[source_distance_column] = df[source_distance_column].astype(float)
    df[distance_column] = df[distance_column].astype(float)

    # df = df[df[source_distance_column] != 0.0]
    if exclude_ids:        
        df = df[~df[near_column].isin(exclude_ids)]
    average_distances = (
        df.groupby([source_column, near_column])
        .agg({distance_column: "mean"})
        .reset_index()
    )
    df.drop(columns=[source_distance_column], inplace=True, errors="ignore")
    average_distances = average_distances.rename(
        columns={distance_column: source_distance_column}
    )

    # Merge this average back into the original DataFrame
    df = merge(df, average_distances, on=[source_column, near_column])
    df[source_distance_column] = round(df[source_distance_column], 3)
    return df

def pipeline_one_off_three_missing(
    number_of_samples: int = -1,
):
    config, logger = initialize()
    dataset_dict, _ = get_dataset_dict_and_df(config)
    keys = list(dataset_dict.keys())
    key_combinations = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for k in range(j + 1, len(keys)):
                key_combinations.append((keys[i], keys[j], keys[k]))
    df = DataFrame(columns=["missing_ref", "average_distance"])

    # Sample x number of key combinations randomly
    keys = (
        random.sample(key_combinations, number_of_samples)
        if number_of_samples > 0
        else key_combinations
    )

    logger.info(
        f"Number of key combinations: {len(keys)}, it may take {0.1*len(keys)} minutes."
    )

    # calculate_average_source_distance(
    #     exclude_ids=keys[1]
    # )
    
    
    from dask.distributed import Client


    logging_config = {
        "version": 1,
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "output.log",
                "level": "INFO",
            },
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
            }
        },
        "loggers": {
            "distributed.worker": {
                "level": "INFO",
                "handlers": ["file", "console"],
            },
            "distributed.scheduler": {
                "level": "INFO",
                "handlers": ["file", "console"],
            }
        }
    }
    dask.config.config['logging'] = logging_config
    # Create a Dask cluster
    client = Client('tcp://dask-scheduler:8786')
    # client.run(initialize_worker)

    # future = client.submit(calculate_average_source_distance, keys[1])

    # # Get the result of the task
    # result = future.result()

    # # Print the result
    # print(result)

    
    future_results = []
    for key in keys[:10]:
        future_results.append(calculate_average_source_distance(
            exclude_ids=key,
            config=config,
        ))

    futures = dask.persist(*future_results)  # trigger computation in the background

    results = dask.compute(*futures)
    print(results[:5])



if __name__ == "__main__":
    pipeline_one_off_three_missing()
