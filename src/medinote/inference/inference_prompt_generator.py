import json
import os
from numpy import nan, ndarray
from openai import AzureOpenAI
from pandas import DataFrame, Series, read_parquet
from medinote import (
    dynamic_load_function,
    merge_parquet_files,
    read_dataframe,
    write_dataframe
)
from medinote.utils.conversion import convert_to_select_all_query
import logging
import requests


logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


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
    from medinote.embedding.vector_search import opensearch_vector_query

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


# def infer(query: str, vector_store=None, config: dict = None):

#     given_schema = config.get("schemas").get("companies")
#     if given_schema:
#         prompt = generate_sql_inference_prompt(query, given_schema)
#     else:
#         prompt = generate_inference_prompt(query, vector_store=vector_store)
#     template = config.get("inference").get("payload_template")
#     payload = template.format(**{"prompt": prompt})

#     inference_url = config.get("inference").get("inference_url")
#     response = generate_via_rest_client(payload=payload, inference_url=inference_url)
#     return response


def parallel_infer_to_delete(df: DataFrame = None, config: dict = None) -> DataFrame:
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


def replace_ilike_with_like(text):
    words = text.split()
    replaced_words = [word if word.lower() != "ilike" else "like" for word in words]
    return " ".join(replaced_words)


def put_columns_with_slash_in_brackets(query: str):
    replacement = r'[\g<0>]'
    query = re.sub(r'(?<![\["\'])\b\w+\/\w+\b(?![\]"\'])', replacement, query)
    return query


def remove_after_and(sql_query):
    # Find the index of 'ORDER BY' in the query
    index = sql_query.upper().find("AND")

    # If 'ORDER BY' is found, return the substring up to that point
    if index != -1:
        return sql_query[:index]

    # If 'ORDER BY' is not found, return the original query
    return sql_query


def remove_order_by(sql_query):
    # Find the index of 'ORDER BY' in the query
    index = sql_query.upper().find("ORDER BY")

    # If 'ORDER BY' is found, return the substring up to that point
    if index != -1:
        return sql_query[:index]

    # If 'ORDER BY' is not found, return the original query
    return sql_query


def merge_all_screened_files(
    pattern: str = None, output_path: str = None, config: dict = None
):
    pattern = pattern or config.get("inference").get("merge_pattern")
    output_path = output_path or config.get("inference").get("merge_output_path")
    df = merge_parquet_files(pattern)
    write_dataframe(df=df, output_path=output_path)


def row_postprocess_sql(row: dict, config: dict):
    sql_names_map = config.get("sql_names_map") or {}
    sql_column = config.get("sql_column")
    if not sql_column:
        raise ValueError("sql_column is required in the config")
    query = row.get(sql_column)
    query = convert_to_select_all_query(query=query)
    for name, value in sql_names_map.items():
        query = query.replace(name, value)
    query = remove_order_by(query)
    query = replace_ilike_with_like(query)
    query = remove_after_and(query)

    # Step 1: Strip whitespace
    query = query.strip()

    # Step 2: Remove the semicolon if present
    if query.endswith(";"):
        query = query[:-1]

    # Step 3: Add 'LIMIT 5'
    query += " LIMIT 5"
    processed_column = config.get("processed_column") or "postprocessed_sql"
    row[processed_column] = query
    return row


import json
import re


def escape_control_characters(json_string):
    # Define a regex pattern for control characters
    control_chars = "".join(
        map(chr, range(0, 32))
    )  # control characters range from 0x00 to 0x1F
    control_chars += chr(127)  # 0x7F is also a control character
    control_chars_regex = re.compile(f"[{re.escape(control_chars)}]")

    # Escape control characters by replacing them with their escaped versions
    escaped_json_string = control_chars_regex.sub(
        lambda match: "\\u{0:04x}".format(ord(match.group())), json_string
    )

    return escaped_json_string


def make_row_element_json_compliant(row: dict):
    for key, value in row.items():
        try:
            json.loads('{"xxx": "{value}"}')
        except:
            row[key] = escape_control_characters(value)

    return row


def apply_postprocess(query: str, level: int, sql_names_map: str = {}):
    if level == 0:
        query = replace_ilike_with_like(query)
    elif level == 1:
        for name, value in sql_names_map.items():
            query = query.replace(name, value)
    elif level == 2:
        query = convert_to_select_all_query(query=query)
    elif level == 3:
        query = remove_order_by(query)
    elif level == 4:
        query = remove_after_and(query)
    # Step 1: Strip whitespace
    query = query.strip()

    # Step 2: Remove the semicolon if present
    if query.endswith(";"):
        query = query[:-1]
    return query


def construct_payload(row: dict, config: dict):
    if isinstance(row, dict) or isinstance(row, Series):
        for key, value in row.items():
            if (
                isinstance(value, list)
                or isinstance(value, tuple)
                or isinstance(value, set)
                or isinstance(value, ndarray)
            ):
                try:
                    row[key] = "\n".join(list(value))
                except Exception as e:
                    logger.warning(
                        f"Ignoring ....Error converting {key} to string: {e}"
                    )

    if prompt_template := config.get("prompt_template"):
        row["prompt"] = prompt_template.format(**row).replace('"', '\\"')

    row = make_row_element_json_compliant(row)

    payload_template = config.get("payload_template")
    payload = payload_template.format(**row)
    try:
        payload = json.loads(payload, strict=False)
    except json.JSONDecodeError as e:
        payload = payload.replace("\n", "\\n")
    payload = json.loads(payload, strict=False) if isinstance(payload, str) else payload
    return payload


def row_azure_openai_infer(row: dict, config: dict):
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        or config.get("azure_openai_endpoint"),
        api_key=os.getenv("AZURE_OPENAI_KEY") or config.get("azure_openai_key"),
        api_version=os.getenv("AZURE_OPENAI_VERSION")
        or config.get("azure_openai_version"),
    )
    instruction = (
        row.get("instruction")
        or config.get("instruction")
        or "You are an AI assistant that helps people find information"
    )
    prompt_column = config.get("prompt_column") or "prompt"

    message_text = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": row.get(prompt_column)},
    ]

    response = client.chat.completions.create(
        model=config.get("model"),  # model = "deployment_name"
        messages=message_text,
        temperature=config.get("temperature") or 0.7,
        max_tokens=config.get("max_tokens") or 800,
        top_p=config.get("top_p") or 0.95,
        frequency_penalty=config.get("frequency_penalty") or 0,
        presence_penalty=config.get("presence_penalty") or 0,
        stop=config.get("stop") or None,
    )
    output = response.choices[0].message.content

    return output


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
    try:
        logger = config.get("logger") or logger
    except Exception as e:
        logger = logging.getLogger("NOIDEA")
    try:
        if isinstance(row, Series) and not (row.notna().all() and row.any()):
            return row
    except Exception as e:
        logger.error(f"Ignoring ....Error checking if row is empty: {e}")

    try:
        if os.getenv("AZURE_OPENAI_ENDPOINT") or config.get("azure_openai_endpoint"):
            result = row_azure_openai_infer(row=row, config=config)
        else:
            inference_url = config.get("inference_url")
            headers = config.get("headers") or {"Content-Type": "application/json"}
            if token := (row.get("token") or config.get("token")):
                headers["Authorization"] = f"Bearer {token}"
            elif token_fucntion := config.get("token_function"):
                token = dynamic_load_function(token_fucntion)()
                headers["Authorization"] = f"Bearer {token}"
            payload = construct_payload(row=row, config=config)

            response = requests.post(url=inference_url, headers=headers, json=payload)
            row["status_code"] = int(response.status_code)
            response.raise_for_status()
            result = response.json()
        if "api_response" in result:
            row["api_response"] = response.json()["api_response"].strip()
    except requests.RequestException as e:
        logger.error(f"Error fetching URL {inference_url}: {e}")
        result = json.dumps({"error": f"Error fetching URL {inference_url}: {e}"})
    except Exception as e:
        logger.error(f"Error on the response:\n \n from {inference_url}:\n {e}")
        result = json.dumps(
            {"error": f"Error on the response:\n \n from {inference_url}:\n {e}"}
        )
    if response_column := (config.get("response_column") or "inference_response"):
        if not isinstance(result, str):
            if "choices" in result and len(result["choices"]) > 0:
                result = result["choices"][0]
            result = (
                result.get(response_column) or result.get("text") or json.dumps(result)
            )
            if isinstance(result, list) and len(result) == 1:
                result = result[0]
        row[response_column] = result
    if "status_code" not in row:
        row["status_code"] = 500
    if embedding_element := row.get("embedding"):
        embedding_json = json.loads(embedding_element)
        if data := embedding_json.get("data"):
            row["embedding"] = (
                data[0].get("embedding") if len(data) > 0 else data.get("embedding")
            )
        if not isinstance(row.get("embedding"), list):
            raise ValueError(f"Invalid embedding found in the row: {row}")
    return row


def parallel_row_infer_to_delete(
    config: dict, df: DataFrame = None, headers: dict = None, persist: bool = False
):
    input_path = config.get("input_path")
    if df is None:
        df = read_dataframe(input_path)
    if df is None or df.empty:
        logger.error(f"Empty DataFrame found at {input_path}")
        return
    df.replace("", nan, inplace=True)
    df = df.dropna().parallel_apply(
        row_infer, axis=1, config=config, headers=headers, **kwargs
    )
    output_path = config.get("output_path")
    if persist and output_path:
        write_dataframe(df, output_path)
    return df
    