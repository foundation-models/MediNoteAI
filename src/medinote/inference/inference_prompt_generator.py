import os
from pandas import DataFrame, Series, read_parquet
from medinote import initialize, merge_parquet_files
from medinote.cached import read_dataframe, write_dataframe
from medinote.curation.rest_clients import generate_via_rest_client
from medinote.embedding.vector_search import opensearch_vector_query

logger, _ = initialize()


def generate_inference_prompt(
    query: str,
    template: str = None,
    id_column: str = None,
    content_column: str = None,
    sql_column: str = None,
    sample_df: DataFrame = None,
    vector_store=None,
    config: dict = None,
) -> str:

    template = template or config.get("inference")["prompt_template"]
    id_column = id_column or config.get("inference")["id_column"]
    content_column = content_column or config.get("inference")["content_column"]
    sql_column = sql_column or config.get("finetune")["sql_column"]
    if not sample_df:
        sample_df = read_parquet(config.get("inference")["sample_df_path"])

    doc_id_list = opensearch_vector_query(
        query, vector_store=vector_store, return_doc_id=True
    )

    row_dict = {"input": query}

    if sample_df is not None:
        if content_column in sample_df.columns:
            subset_df = sample_df[sample_df[id_column].isin(doc_id_list)]
            samples = [
                "input: "
                + str(row[content_column])
                + "\t output:"
                + str(row[sql_column])
                for _, row in subset_df.iterrows()
            ]
            samples = [sample for sample in samples if query not in sample]
            samples = "\n".join(samples)
            row_dict["samples"] = samples
    else:
        logger.error(
            f"Invalid value for sample_df or id_column: {sample_df}, {id_column}"
        )
    template = template or config.get("finetune")["prompt_template"]
    logger.debug(f"Using template: {template}")
    prompt = template.format(**row_dict)
    logger.debug(f"Prompt: {prompt}")
    return prompt


def generate_sql_inference_prompt(
    query: str, sql_schema: str, template: str = None, config: dict = None
) -> str:

    template = template or config.get("sqlcoder")["prompt_template"]
    row_dict = {"question": query, "ddl": sql_schema}
    prompt = template.format(**row_dict)
    logger.debug(f"Prompt: {prompt}")
    return prompt


def infer_for_dataframe(
    row: Series,
    input_column: str = None,
    output_column: str = None,
    vector_store=None,
    config: dict = None,
):
    row[output_column] = infer(
        row[input_column], vector_store=vector_store, config=config
    )
    return row


def infer(query: str, vector_store=None, config: dict = None):

    given_schema = config.get("schemas").get("dealcloud_provider_fs_companies_a")
    if given_schema:
        prompt = generate_sql_inference_prompt(query, given_schema)
    else:
        prompt = generate_inference_prompt(query, vector_store=vector_store)
    template = config.get("inference").get("payload_template")
    payload = template.format(**{"prompt": prompt})

    inference_url = config.get("inference").get("inference_url")
    response = generate_via_rest_client(payload=payload, inference_url=inference_url)
    return response


def parallel_infer(df: DataFrame = None, config: dict = None) -> DataFrame:
    df = df or read_parquet(config.get("inference")["input_path"])
    response_column = config.get("inference")["response_column"]
    query_column = config.get("inference")["query_column"]
    output_path = config.get("inference")["output_path"]
    # vector_store = get_vector_store()

    chunk_size = 10
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df))
        chunk_df = df[start_index:end_index]

        output_file = (
            f"{output_path}_{start_index}_{end_index}.parquet" if output_path else None
        )
        if output_file is None or not os.path.exists(output_file):
            chunk_df = chunk_df.parallel_apply(
                infer_for_dataframe,
                axis=1,
                input_column=query_column,
                output_column=response_column,
                #    vector_store=vector_store, # has problem hangs
            )

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


def merge_all_screened_files(
    pattern: str = None, output_path: str = None, config: dict = None
):
    pattern = pattern or config.get("inference").get("merge_pattern")
    output_path = output_path or config.get("inference").get("merge_output_path")
    df = merge_parquet_files(pattern)
    write_dataframe(df=df, output_path=output_path)


def row_infer(row: dict, config: dict):
    """
    Perform inference on a single row of data using the provided configuration.

    Args:
        row (dict): A dictionary representing a single row of data.
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        dict: The updated row dictionary with the inference result added.

    Raises:
        requests.exceptions.RequestException: If there is an error making the inference request.

    """
    import requests
    import json

    template = config.get("prompt_template")
    prompt = template.format(**row)
    prompt_column = config.get("prompt_column")
    if prompt_column:
        row[prompt_column] = prompt

    template = config.get("payload_template")
    payload = template.format(**{"prompt": prompt})
    payload = payload.replace("\n", "\\n")
    payload = json.loads(payload)

    inference_url = config.get("inference_url")
    headers = {"Content-Type": "application/json"}
    response = requests.post(url=inference_url, headers=headers, json=payload)
    result = response.json()
    response_column = config.get("response_column") or "inference_response"
    if response_column:
        row[response_column] = (
            result["text"] if "text" in result else json.dumps(result)
        )
    return row


def parallel_row_infer(config: dict, df: DataFrame = None, persist: bool = False):
    input_path = config.get("input_path")
    if df is None:
        df = read_dataframe(input_path)
    if df is None or df.empty:
        logger.error(f"Empty DataFrame found at {input_path}")
        return
    df = df.parallel_apply(row_infer, axis=1, config=config)
    output_path = config.get("output_path")
    if persist and output_path:
        write_dataframe(df, output_path)
    return df


if __name__ == "__main__":
    merge_all_screened_files()
    # infer("Find all assets in San Fancisco with a value greater than 100000")
