import os
from pandas import DataFrame, Series, read_parquet
from medinote import initialize, merge_parquet_files
from medinote.curation.rest_clients import generate_via_rest_client
from medinote.embedding.vector_search import get_vector_store, opensearch_vector_query


config, logger = initialize()


def generate_inference_prompt(query: str,
                              template: str = None,
                              id_column: str = None,
                              content_column: str = None,
                              sql_column: str = None,
                              sample_df: DataFrame = None,
                              vector_store = None,
                              ) -> str:

    template = template or config.inference['prompt_template']
    id_column = id_column or config.inference['id_column']
    content_column = content_column or config.inference['content_column']
    sql_column = sql_column or config.finetune['sql_column']
    if not sample_df:
        sample_df = read_parquet(config.inference['sample_df_path'])

    doc_id_list = opensearch_vector_query(query,
                                          vector_store=vector_store,
                                          return_doc_id=True)

    row_dict = {"input": query}

    if sample_df is not None:
        if content_column in sample_df.columns:
            subset_df = sample_df[sample_df[id_column].isin(doc_id_list)]
            samples = [
                "input: " + str(row[content_column]) +
                "\t output:" + str(row[sql_column])
                for _, row in subset_df.iterrows()
            ]
            samples = [sample for sample in samples if query not in sample]
            samples = '\n'.join(samples)
            row_dict["samples"] = samples
    else:
        logger.error(
            f"Invalid value for sample_df or id_column: {sample_df}, {id_column}")
    template = template or config.finetune['prompt_template']
    logger.debug(f"Using template: {template}")
    prompt = template.format(**row_dict)
    logger.debug(f"Prompt: {prompt}")
    return prompt


def infer_for_dataframe(row: Series,
                        input_column: str = None,
                        output_column: str = None,
                        vector_store = None
                        ):
    row[output_column] = infer(row[input_column], vector_store=vector_store)
    return row


def infer(query: str, vector_store = None):
    prompt = generate_inference_prompt(query, vector_store=vector_store)
    template = config.inference.get('payload_template')
    payload = template.format(**{"prompt": prompt})

    inference_url = config.inference.get('inference_url')
    response = generate_via_rest_client(payload=payload,
                                        inference_url=inference_url
                                        )
    return response


def parallel_infer(df: DataFrame = None,
                   ) -> DataFrame:
    df = df or read_parquet(config.inference['input_path'])
    response_column = config.inference['response_column']
    query_column = config.inference['query_column']
    output_path = config.inference['output_path']
    # vector_store = get_vector_store()

    chunk_size = 10
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df))
        chunk_df = df[start_index:end_index]
        
        output_file = f"{output_path}_{start_index}_{end_index}.parquet" if output_path else None
        if output_file is None or not os.path.exists(output_file):   
            chunk_df = chunk_df.parallel_apply(infer_for_dataframe, axis=1,
                                                input_column=query_column,
                                                output_column=response_column,
                                            #    vector_store=vector_store, # has problem hangs
                                                )        

            if output_file:
                try:
                    chunk_df.to_parquet(output_file)
                except Exception as e:
                    logger.error(
                        f"Error saving the embeddings to {output_file}: {repr(e)}")
        else:
            logger.info(
                f"Skipping chunk {start_index} to {end_index} as it already exists.")


def merge_all_screened_files(pattern: str = None, 
                             output_path: str = None):
    pattern = pattern or config.inference.get('merge_pattern')
    output_path = output_path or config.inference.get('merge_output_path')
    df = merge_parquet_files(pattern)
    df.to_parquet(output_path)


if __name__ == "__main__":
    merge_all_screened_files()
    # infer("Find all assets in San Fancisco with a value greater than 100000")
