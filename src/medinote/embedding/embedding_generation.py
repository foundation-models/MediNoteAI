from pandas import DataFrame, read_parquet
from medinote import initialize
import hashlib


config, logger = initialize()

def retrive_embedding(query: str,
              embedding_function: callable = None,
              ):
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
              embedding_function: callable = None,
              index_column: str = 'doc_id',
              column2embed: str = 'embedding',
              ):
    row[index_column], row[column2embed] = retrive_embedding(
        query=row[column_name],
        embedding_function=embedding_function,
    )
    return row


def parallel_generate_embedding(df: DataFrame = None,
                    column_name: str = None,
                    ):

    input_path = config.embedding.get('input_path')
    if not df and input_path:
        logger.debug(f"Reading the input parquet file from {input_path}")
        df = read_parquet(input_path)
        # df = df[:100]

    column_name = column_name or config.embedding.get('column2embed')

    # Apply embed_row function to each row of the DataFrame in parallel.
    # The result is a Series with lists of embeddings.
    logger.debug(f"Applying the embedding function to the DataFrame")
    embedded_df = df.parallel_apply(generate_embedding, axis=1,
                                    column_name=column_name,
                                    )

    # Remove None entries
    embedded_df = embedded_df.dropna()

    output_path = config.embedding.get('output_path')
    if output_path:
        logger.debug(f"Saving the embeddings to {output_path}")
        embedded_df.to_parquet(output_path)

    return embedded_df

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
