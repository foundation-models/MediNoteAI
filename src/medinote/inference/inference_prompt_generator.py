import os
from pandas import DataFrame, Series, read_parquet
from medinote import initialize
from medinote.curation.rest_clients import generate_via_rest_client
from medinote.embedding.vector_search import opensearch_vector_query


config, logger = initialize()


def generate_inference_prompt(query: str,
                              template: str = None,
                              id_column: str = None,
                              content_column: str = None,
                              sql_column: str = None,
                              sample_df: DataFrame = None,
                              ) -> str:

    template = template or config.inference['prompt_template']
    id_column = id_column or config.inference['id_column']
    content_column = content_column or config.inference['content_column']
    sql_column = sql_column or config.finetune['sql_column']
    if not sample_df:
        sample_df = read_parquet(config.inference['sample_df_path'])

    doc_id_list = opensearch_vector_query(query, 
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


def infer(query: str):
    prompt = generate_inference_prompt(query)
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
    
    chunk_size = 10
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df))
        chunk_df = df[start_index:end_index]
        output_file = f"{output_path}_{start_index}_{end_index}.parquet"
        if not os.path.exists(output_file):
            chunk_df[response_column] = chunk_df[query_column].parallel_apply(infer)
            chunk_df.to_parquet(output_file)
        else:
            logger.info(f"Skipping chunk {start_index} to {end_index} as it already exists.")
    return df


if __name__ == "__main__":
    parallel_infer()
    # infer("Find all assets in San Fancisco with a value greater than 100000")
