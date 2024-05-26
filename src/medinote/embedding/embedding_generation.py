from medinote import dynamic_load_function_from_env_varaibale_or_config, initialize
import hashlib
import logging
import os

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


def retrieve_embedding(query: str,
                       config: dict=None,
                       inference_url: str=None,
                       ):
    """
    Retrieves the embedding for a given query.

    Args:
        query (str): The input query for which the embedding needs to be retrieved.

    Returns:
        tuple: A tuple containing the document ID (hash of the query) and the corresponding embedding.
    """
    inference_url = inference_url or config.get('embedding_url')
    try:
        payload = {"input": [query]}
        embedding_function = dynamic_load_function_from_env_varaibale_or_config(
            key="embedding_function",
            config=config,
            default_function="medinote.curation.rest_clients.generate_via_rest_client"
        )
        embeddings = embedding_function(
            payload=payload, inference_url=inference_url
        )
        # Create a hash of query as the ID
        doc_id = hashlib.sha256(query.encode()).hexdigest()
        return doc_id, embeddings[0]
    except Exception as e:
        logger.error(f"Error embedding row: {repr(e)}")
        raise e


def generate_embedding(
    row: dict,
    config: dict=None,
):
    index_column = config.get("index_column")
    embedding_column = config.get("embedding_column")
    column2embed = config.get("column2embed")
    row[index_column], row[embedding_column] = retrieve_embedding(
        query=row[column2embed],
        config=config,
    )
    return row

# def parallel_generate_embedding(
#     df: DataFrame = None,
#     column_name: str = None,
#     config: dict = None,
#     index_column: str = None,
#     embedding_column: str = None,
#     persist: bool = True,
# ):
#     """
#     Generate embeddings for each row of a DataFrame in parallel.

#     Args:
#         df (DataFrame, optional): The input DataFrame. If not provided, it will be read from the input_path specified in the configuration.
#         column_name (str, optional): The name of the column to generate embeddings for. If not provided, it will be obtained from the configuration.

#     Returns:
#         None

#     Raises:
#         None
#     """

#     if df is None:
#         input_path = config.get("input_path")
#         if not input_path:
#             raise ValueError(f"No input_path found.")
#         logger.debug(f"Reading the input parquet file from {input_path}")
#         df = read_parquet(input_path)
#         # df = df[:100]

#     column_name = column_name or config.get("column2embed")

#     # Apply embed_row function to each row of the DataFrame in parallel.
#     # The result is a Series with lists of embeddings.
#     logger.debug(f"Applying the embedding function to the DataFrame")

#     chunk_size = 1000
#     num_chunks = len(df) // chunk_size + 1
    
#     output_prefix = config.get("output_prefix")

#     chunk_df_list = []
#     for i in range(num_chunks):
#         start_index = i * chunk_size
#         end_index = min((i + 1) * chunk_size, len(df))
#         chunk_df = df[start_index:end_index]

#         output_chunk_file = (
#             f"{output_prefix}_{start_index}_{end_index}.parquet"
#             if output_prefix
#             else None
#         )
#         if not os.path.exists(output_chunk_file):
#             # chunk_df = chunk_df.astype(str)
#             try:
#                 chunk_df = chunk_df.parallel_apply(
#                     generate_embedding,
#                     axis=1,
#                     column_name=column_name,
#                     config=config,
#                     index_column=index_column,
#                     embedding_column=embedding_column,
#                 )
#                 chunk_df_list.append(chunk_df)
#             except ValueError as e:
#                 if "Number of processes must be at least 1" in str(e):
#                     logger.error(
#                         f"No idea for error: Number of processes must be at least \n ignoring ....."
#                     )
#             except Exception as e:
#                 logger.error(f"Error generating synthetic data: {repr(e)}")
#             logger.debug(f"Saving the embeddings to {output_chunk_file}")
#             if persist:
#                 try:
#                     chunk_df.to_parquet(output_chunk_file)
#                 except Exception as e:
#                     logger.error(
#                         f"Error saving to {output_chunk_file}: {repr(e)}"
#                     )
#         else:
#             logger.info(
#                 f"Skipping chunk {start_index} to {end_index} as it already exists."
#             )
#     if persist:
#         pattern = f"{output_prefix}_*.parquet"
#         output_path = config.get("output_path") or f"{output_prefix}.parquet"
#         column_names_map = config.get("column_names_map")
#         merged_df = merge_all_chunks(
#             pattern=pattern, output_path=output_path, column_names_map=column_names_map
#         )
#     else:
#         merged_df = concat(chunk_df_list, ignore_index=True)

