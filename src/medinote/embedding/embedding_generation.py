import os
from pandas import DataFrame, read_parquet
from medinote import dynamic_load_function_from_env_varaibale_or_config, initialize
import hashlib


config, logger = initialize()

embedding_function = dynamic_load_function_from_env_varaibale_or_config(
    'embedding_function')


def retrive_embedding(query: str):
    try:
        payload = {
            "input": [query]
        }
        embeddings = embedding_function(payload=payload,
                                        inference_url=config.embedding.get(
                                            'embedding_url')
                                        )
        # Create a hash of query as the ID
        doc_id = hashlib.sha256(query.encode()).hexdigest()
        return doc_id, embeddings
    except Exception as e:
        logger.error(f"Error embedding row: {repr(e)}")
        raise e


def generate_embedding(row: dict,
                       column_name: str,
                       index_column: str = 'doc_id',
                       column2embed: str = 'embedding',
                       ):
    row[index_column], row[column2embed] = retrive_embedding(
        query=row[column_name],
    )
    return row


def parallel_generate_embedding(df: DataFrame = None,
                                column_name: str = None,
                                ):

    input_path = config.embedding.get('input_path')
    output_path = config.embedding.get('output_path')
    if not df and input_path:
        logger.debug(f"Reading the input parquet file from {input_path}")
        df = read_parquet(input_path)
        # df = df[:100]

    column_name = column_name or config.embedding.get('column2embed')

    # Apply embed_row function to each row of the DataFrame in parallel.
    # The result is a Series with lists of embeddings.
    logger.debug(f"Applying the embedding function to the DataFrame")

    chunk_size = 1000
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df))
        chunk_df = df[start_index:end_index]

        output_file = f"{output_path}_{start_index}_{end_index}.parquet" if output_path else None
        if output_file is None or not os.path.exists(output_file):
            chunk_df = chunk_df.astype(str)
            chunk_df = chunk_df.parallel_apply(generate_embedding, axis=1,
                                                column_name=column_name,
                                                )
            logger.debug(f"Saving the embeddings to {output_file}")
            if output_file:
                try:
                    chunk_df.to_parquet(output_file)
                except Exception as e:
                    logger.error(
                        f"Error saving the embeddings to {output_file}: {repr(e)}")
        else:
            logger.info(
                f"Skipping chunk {start_index} to {end_index} as it already exists.")


if __name__ == "__main__":
    parallel_generate_embedding()

# def xxxx(df: DataFrame,
#                               vector_index: IndexType,
#                               column2embed: str = None,
#                               ):
#     try:
#         column2embed = column2embed or config.embedding.get(
#             'column2embed')
#         documents = df.parallel_apply(generate_embedding,
#                                       column_name=column2embed,
#                                       axis=1).tolist()
#         for d in documents:
#             vector_index.insert(document=d)
#     except Exception as e:
#         logger.error(
#             f"***************************  Exception: {repr(e)} *****************")
