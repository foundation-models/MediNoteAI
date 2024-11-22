import json
import os
import re
from functools import cache
from hashlib import md5
from typing import Any

import psycopg2
import psycopg2.pool

from dotenv import load_dotenv
from medinote import chunk_process, initialize, write_dataframe
from pandas import DataFrame, concat, merge, read_parquet
from medinote.inference.inference_prompt_generator import row_infer
from opentelemetry import trace


main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
)

pgvector_connection_config = main_config.get("pgvector_connection")


# Initialize the connection pool
connection_pool = psycopg2.pool.SimpleConnectionPool(
    1,
    20,
    user=pgvector_connection_config.get("user"),
    password=pgvector_connection_config.get("password"),
    host=pgvector_connection_config.get("host"),
    port=pgvector_connection_config.get("port"),
    database=pgvector_connection_config.get("database"),
)


def __get_pgvector_table_name(config):
    env_prefix = "demo_" if (os.environ.get("ENV") or "").lower() == "demo" else ""
    pgvector_table_name = config.get("pgvector_table_name")
    return f"{env_prefix}{pgvector_table_name}"


def get_connection():
    try:
        connection = connection_pool.getconn()
        if connection:
            return connection
        else:
            connection = psycopg2.connect(
                user=pgvector_connection_config.get("user"),
                password=pgvector_connection_config.get("password"),
                host=pgvector_connection_config.get("host"),
                port=pgvector_connection_config.get("port"),
                database=pgvector_connection_config.get("database"),
            )
            return connection
    except Exception as e:
        logger.error(f"Error getting connection: {e}")
        return None


def release_connection(connection):
    try:
        connection_pool.putconn(connection)
    except Exception as e:
        logger.error(f"Error releasing connection: {e}")


def close_connection(connection):
    try:
        connection.close()
    except Exception as e:
        logger.error(f"Error closing connection: {e}")


def execute_query(query, params=None):
    connection = None
    cursor = None
    result = []
    column_names = []
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(query, params)
        if cursor.description:
            column_names = [desc[0] for desc in cursor.description]  # Fetch column names
        else:
            column_names = []
        try:
            result = cursor.fetchall()
        except psycopg2.ProgrammingError as e:
            if "no results to fetch" in str(e):
                pass
            else:
                raise e
        connection.commit()
    except Exception as e:
        logger.error("Error executing query", query, e)
        if connection:
            connection.rollback()
        if cursor:
            cursor.close()
        if connection:
            close_connection(connection)
    finally:
        if cursor:
            cursor.close()
        if connection:
            release_connection(connection)
    return result, column_names


def calculate_average_source_distance(
    df: DataFrame = None,
    source_column: str = "source_ref",
    near_column: str = "near_ref",
    distance_column: str = "distance",
    source_distance_column: str = "source_distance",
    exclude_ids=None,
    persist: bool = True,
    config: dict = None,
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
        output_path = config.get("cross_distance_output_path")
        if output_path:
            df = read_parquet(output_path)

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
    output_path = config.get("cross_document_distance_output_path")
    df[source_distance_column] = round(df[source_distance_column], 3)
    if persist and output_path:
        write_dataframe(df=df, output_path=output_path)
    return df


def get_dataset_dict_and_df(config):
    dataset_parquet_path = config.get("dataset_parquet_path") if config else None
    if not dataset_parquet_path:
        raise ValueError(
            "You must provide either a dataset_dict or a dataset_parquet_file"
        )
    dataset_df = read_parquet(dataset_parquet_path)
    id_column = config.get("id_column", "doc_id") if config else "doc_id"
    content_column = config.get("embedding_column", "content") if config else "content"
    dataset_dict = dataset_df.set_index(id_column)[content_column].to_dict()

    return dataset_dict, dataset_df


def search_by_natural_language(
    query: str,
    config: dict,
    data_provider_id: int,
    client_id: int = None,
    embedding: list = None,
    condition: str = None
):
    if embedding is None:
        embedding = get_embedding(query, config)

    if len(embedding) == 0:
        raise ValueError("Invalid query vector")

    if not isinstance(embedding, list):
        embedding = embedding.tolist()

    df = search_by_vector(
        query_vector=embedding,
        config=config,
        condition=condition,
        data_provider_id=data_provider_id,
        client_id=client_id,
        threshold=config["threshold"]
    )
    duplicate_column_to_check = config.get("duplicate_column_to_check")
    if duplicate_column_to_check:
        df.drop_duplicates(duplicate_column_to_check, inplace=True)
    max_results = config.get("max_results")
    if max_results:
        df = df[:max_results]
    return df, embedding


def get_embedding(query, config):
    row = {"query": query}
    if instruct := config.get("embedding_instruct"):
        row["embedding_input"] = "Instruct: " + instruct + "\nQuery: " + query
    else:
        row["embedding_input"] = query
    row = row_infer(row=row, config=config)
    embedding = row.get("embedding")
    return embedding


def dict_to_hashable(d):
    # Recursively convert dictionary to a tuple of tuples
    def make_hashable(obj):
        if isinstance(obj, dict):
            # Recursively convert each value in the dictionary
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            # Convert lists to tuples and recursively convert each element
            return tuple(make_hashable(i) for i in obj)
        elif isinstance(obj, set):
            # Convert sets to frozensets
            return frozenset(make_hashable(i) for i in obj)
        else:
            # Return the object if it is already hashable
            return obj

    return make_hashable(d)


def hashable_to_dict(hashable):
    # Recursively convert hashable object back to dictionary
    def make_dict(obj):
        if isinstance(obj, tuple) and all(
            isinstance(i, tuple) and len(i) == 2 for i in obj
        ):
            # Convert tuple of tuples back to dictionary
            return {k: make_dict(v) for k, v in obj}
        elif isinstance(obj, tuple):
            # Convert tuple back to list
            return [make_dict(i) for i in obj]
        elif isinstance(obj, frozenset):
            # Convert frozenset back to set
            return {make_dict(i) for i in obj}
        else:
            # Return the object if it is already hashable
            return obj

    return make_dict(hashable)


def search_by_vector(
    query_vector,
    config: dict,
    data_provider_id: int,
    threshold: float,
    client_id: int = None,
    condition: str = None
):
    with trace.get_tracer(__name__).start_as_current_span("Search by vector") as span:
        limit = config.get("limit", 10)

        include_row_keys = config.get("include_row_keys", [])
        pgvector_table_name = __get_pgvector_table_name(config)
        embedding_column = config.get("embedding_column") or "embedding"
        embedding_meta_column = config.get("embedding_meta_column") or "meta"

        if not pgvector_table_name:
            raise ValueError("pgvector_table_name not found in config")

        select_keys_str = ", ".join(map(lambda key: f"{embedding_meta_column}->>'{key}' AS {key}", include_row_keys))
        select_keys_formatted_str = select_keys_str or "NULL as meta"

        and_condition = f" AND {condition}" if condition else ""
        and_condition += f" AND client_id = {client_id}" if client_id is not None else ""

        select_columns = config.get("select_columns", include_row_keys)

        query = f"""
            SELECT {'DISTINCT' if config.get('use_distinct') else ''}
              {', '.join(select_columns)}
            FROM (
                SELECT
                    {select_keys_formatted_str},
                    {embedding_column},
                    {embedding_column} <-> '{query_vector}' AS distance,
                    ROW_NUMBER() OVER (PARTITION BY meta->>'table_name', meta->>'column_name' ORDER BY {embedding_column} <-> '{query_vector}') AS row_id
                FROM {pgvector_table_name}
                WHERE (meta->>'data_provider_id')::int = {data_provider_id} {and_condition}
                ORDER BY distance
            )
            WHERE row_id < 6 AND {embedding_column} <-> '{query_vector}' < {threshold}
            LIMIT {limit}
        """

        data , _ = execute_query(query)
        df = DataFrame(data, columns=select_columns)

        span.set_attributes({
            "select_table": pgvector_table_name
        })
        return df


def get_data_from_table(config: dict):
    include_row_keys = config.get("include_row_keys", [])
    pgvector_table_name = config.get("pgvector_table_name")
    meta_column = config.get("meta_column") or "meta"

    if not pgvector_table_name:
            raise ValueError("pgvector_table_name is not found in config")

    use_distinct = config.get("use_distinct")

    select_keys_str = ", ".join(map(lambda key: f"{meta_column}->>'{key}' AS {key}", include_row_keys))
    select_keys_formatted_str = select_keys_str or "NULL as meta"

    command = f"""
        SELECT {'DISTINCT' if use_distinct else ''}
            {select_keys_formatted_str}
        FROM {pgvector_table_name}
    """
    
    data = execute_query(command)
    df = DataFrame(data, columns=include_row_keys)

    return df


def search_vector_by_id(vector_id: int, config: dict):
    limit = config.get("limit", 10)
    pgvector_table_name = __get_pgvector_table_name(config)
    embedding_column = config.get("embedding_column") or "embedding"
    embedding_meta_column = config.get("embedding_meta_column") or "meta"

    if not pgvector_table_name:
        raise ValueError("pgvector_table_name not found in config")

    data , _ = execute_query(f"""
        SELECT id, client_id, {embedding_column}, {embedding_meta_column}
        FROM {pgvector_table_name}
        WHERE id = {vector_id}
        LIMIT {limit}
    """)

    return vector_record_to_dict(data[0] if len(data) > 0 else None)


def extract_document(
    row: dict,
    column_name: str,
    index_column: str = "doc_id",
    embedding_column: str = "embedding",
):
    try:
        from llama_index import Document
    except ImportError:
        from llama_index.core import Document
    try:
        return Document(
            text=row[column_name],
            doc_id=row[index_column],
            embedding=row[embedding_column].tolist(),
        )
    except Exception as e:
        logger.error(f"Error embedding row: {repr(e)}")
        raise e


def chunk_documents(documents, chunk_size):
    """Divide documents into chunks based on length."""
    chunks = []
    current_chunk = []

    for doc in documents:
        if len(current_chunk) < chunk_size:
            current_chunk.append(doc)
        else:
            chunks.append(current_chunk)
            current_chunk = [doc]

    # Add the last chunk if it contains any documents
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


@cache
def get_pgvector_connection(hashable_config: tuple = None):
    config = hashable_to_dict(hashable_config) if hashable_config else {}

    load_dotenv()
    dbname = config.get("database") or os.getenv("DB_NAME")
    db_config = {
        "dbname": dbname,
        "user": config.get("user") or os.getenv("DB_USER"),
        "password": config.get("password") or os.getenv("DB_PASSWORD"),
        "host": config.get("host") or os.getenv("DB_HOST") or "localhost",
        "port": config.get("port") or os.getenv("DB_PORT") or 6432,
    }
    try:
        conn = psycopg2.connect(**db_config)
    except Exception as e:
        db_config["dbname"] = "postgres"
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE {dbname}")
        db_config["dbname"] = dbname
        conn = psycopg2.connect(**db_config)
    return conn


def insert_user_question_and_sql_into_vector_database(df, config):
    if config.get("recreate"):
        create_pgvector_table(config=config)
    
    embedding_instruct = config.get("embedding_instruct")
    config["embedding_instruct"] = None
    if "sql" in df.columns:
        df.loc[:, "embedding"] = df["sql"].apply(lambda x: get_embedding(x, config))
        df.loc[:, "embedded_field"] = "sql"
    command = construct_insert_command_with_meta_info(df=df, config=config)
    sql_vectors , _ = execute_query(command) if command else []
    df.loc[:, "embedding"] = df["user_question"].apply(lambda x: get_embedding(x, config))
    df.loc[:, "embedded_field"] = "user_question"
    command = construct_insert_command_with_meta_info(df=df, config=config)
    question_vectors , _ = execute_query(command) if command else []
    config["embedding_instruct"] = embedding_instruct
    return sql_vectors, question_vectors


def insert_into_vector_database_with_meta_info(df, config):
    if config.get("recreate"):
        create_pgvector_table(config=config)
    command = construct_insert_command_with_meta_info(df=df, config=config)
    execute_query(command)
    return df


def construct_update_meta_command(doc_id: int, meta_key: str, meta_value: Any, table_name: str, meta_column: str = "meta"):
    update_query = f"""
        UPDATE {table_name}
        SET {meta_column} = jsonb_set({meta_column}, '{{{meta_key}}}', %s::jsonb)
        WHERE id = %s;
    """
    params = (json.dumps(meta_value), doc_id)
    return update_query, params


def update_feedback(config: dict, doc_id: int, feedback_value: int):
    table_name = __get_pgvector_table_name(config)
    meta_key = 'feedback'

    update_query, params = construct_update_meta_command(doc_id, meta_key, feedback_value, table_name)

    try:
        execute_query(update_query, params)
        return True
    except Exception as e:
        logger.error(f"Failed to update feedback for doc_id {doc_id}: {e}")
        return False


def create_pgvector_table(
    config: dict,
    columns_with_types: dict = None
):
    embedding_column = config.get("embedding_column") or "embedding"
    embedding_meta_column = config.get("embedding_meta_column") or "meta"
    vector_dimension = config.get("vector_dimension") or 1024
    pgvector_table_name = __get_pgvector_table_name(config)
    connection_config = main_config.get("pgvector_connection")

    conn = get_pgvector_connection(hashable_config=dict_to_hashable(connection_config))
    cur = conn.cursor()

    columns_definition = []

    if columns_with_types:
        for column_name, column_type in columns_with_types.items():
            columns_definition.append(f"{column_name} {column_type}")
    else:
        columns_definition = [f"{embedding_column} VECTOR({vector_dimension})", f"{embedding_meta_column} JSONB"]
    columns_definition_str = ", ".join(columns_definition)

    cur.execute(f"""
        DROP TABLE IF EXISTS {pgvector_table_name}
    """)
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {pgvector_table_name} (
            id SERIAL PRIMARY KEY,
            client_id BIGINT NOT NULL DEFAULT 0,
            {columns_definition_str}
        )
    """)
    conn.commit()


def construct_insert_command_with_meta_info(df: DataFrame, config: dict):
    pgvector_table_name = __get_pgvector_table_name(config)
    embedding_column = config.get("embedding_column") or "embedding"
    embedding_meta_column = config.get("embedding_meta_column") or "meta"
    include_row_keys = config.get("include_row_keys")
    client_id = config.get("client_id")
    insert_value_list = []
    meta_values = {}

    for _, row in df.iterrows():
        if embedding_column:
            embedding_value = row.get(embedding_column)
            insert_value = (
                [embedding_value]
                if isinstance(embedding_value, list)
                else [embedding_value.tolist()]
            ) if embedding_value is not None else None
        else:
            insert_value = None
        meta_values = {}
        for key in include_row_keys:
            if key:
                value = row.get(key)
                if isinstance(row.get(key), str):
                    value = value.replace("'", "''")
                meta_values[key] = value
            else:
                include_row_keys.pop(include_row_keys.index(key))

        if insert_value is not None:
            insert_value = ", ".join(f"'{value}'" for value in insert_value)
            insert_meta_value = f"'{json.dumps(meta_values)}'"
            insert_value_list.append(f"({client_id}, {insert_value}, {insert_meta_value})")

    insert_value_list_str = ", ".join(insert_value_list)

    command = f"""
        INSERT INTO {pgvector_table_name} (client_id, {embedding_column}, {embedding_meta_column}) 
        VALUES {insert_value_list_str}
        RETURNING id, client_id, {embedding_column}, {embedding_meta_column}
    """ if insert_value_list else None 

    return command


def construct_insert_command(df: DataFrame, config: dict, columns_with_types: dict = None):
    pgvector_table_name = __get_pgvector_table_name(config)
    include_row_keys = list(columns_with_types.keys()) or config.get("include_row_keys")
    insert_value_list = []

    for _, row in df.iterrows():
        insert_value = []
        for key in include_row_keys:
            if row.get(key):
                value = row.get(key)
                if isinstance(row.get(key), str):
                    value = value.replace("'", "''")
                insert_value.append(value)
            else:
                insert_value.append(None)

        if insert_value:
            insert_value = ", ".join(f"'{value}'" for value in insert_value)
            insert_value_list.append(f"({insert_value})")

    insert_value_list_str = ", ".join(insert_value_list)
    include_row_keys_str = ", ".join(include_row_keys)

    command = f"""
        INSERT INTO {pgvector_table_name} ({include_row_keys_str}) 
        VALUES {insert_value_list_str}
        RETURNING id, client_id, {include_row_keys_str}
    """ if insert_value_list else None 

    return command


def create_report(df: DataFrame, config: dict, columns_with_types: dict):
    if config.get("recreate"):
        create_pgvector_table(config=config, columns_with_types=columns_with_types)

    command = construct_insert_command(df, config, columns_with_types)
    execute_query(command)
    return df


def set_additional_instructions(row: dict, config: dict, data_provider_id: int):
    user_question = row.get("user_question")
    if (this_config:=config.get("additional_instruction_embedding_api")) and this_config.get("max_results") != 0:  
        embedding_df, _ = search_by_natural_language(
            query=user_question,
            config=this_config,
            embedding=row.get("embedding"),
            data_provider_id=data_provider_id
        )
        if threshold := this_config.get("threshold"):
            embedding_df = embedding_df[embedding_df["distance"] < threshold]

    embedding_df["user_question"] = user_question
    row["additional_instructions"] = construct_additional_instructions(
        embedding_df, config
    )

    return row


def generate_sqlcoder_prompt_parameters(message_content_dict: dict, config: dict):
    with trace.get_tracer(__name__).start_as_current_span("Generate sqlcoder prompt parameters"):
        original_query = message_content_dict.get("query", "")
        incorrect_sql = message_content_dict.get("sql", "")
        user_question = message_content_dict.get("user_question", "")
        result_str = message_content_dict.get("result", "{}")
        use_examples = config.get("use_examples", True)
        result = json.loads(result_str)
        reason = result.get("error", "") if result else ""
        row = {}

        (embedding_df, df_history, _) = extract_similar_vectors(message_content_dict, config)

        if use_examples:
            samples = ""
            for _, row_history in df_history.iterrows():
                sql = row_history.get("sql")
                user_question = row_history["user_question"]
                formatted_sql = " ".join(sql) if isinstance(sql, list) else sql
                samples += f"User Question: {user_question} -- converted to SQL: {formatted_sql} \n"
            row["good_samples_prompt"] = (
                f"Here are some examples of correct SQL query generations: \n {samples} \n"
                if samples
                else ""
            )
            row["bad_samples_prompt"] = (
                f"Here are some examples of incorrect SQL query generations: \n {incorrect_sql} \n"
                if incorrect_sql
                else ""
            )
        else:
            row["good_samples_prompt"] = ""
            row["bad_samples_prompt"] = ""

        row["user_question"] = original_query or user_question
        row["past_failure_reasons"] = reason if reason else ""
        row["data_provider_id"] = config.get("data_provider_id")

        table_names = config.get("table_names")
        if config.get("detect_create_table_statement"):
            row["create_table_statements"], df_table_columns = \
                detect_create_table_statement(
                    row,
                    table_names,
                    config["create_table_statement_detection"])
            embedding_df = merge(embedding_df, df_table_columns[["table_name", "column_name"]], on=["table_name", "column_name"], how="inner")
        if not config.get("detect_create_table_statement") or not row["create_table_statements"]:
            row["create_table_statements"] = config.get(
                "create_table_statements"
            ) or message_content_dict.get("create_table_statements", "")
        
        embedding_df = merge(embedding_df, df_table_columns[["table_name", "column_name"]], on=["table_name", "column_name"], how="inner")
        row["additional_instructions"] = construct_additional_instructions(embedding_df, config)

        row["insert_statements"] = (
            message_content_dict.get("insert_statements")
            or config.get("insert_statements")
            or ""
        )
        return row


def extract_similar_vectors(message_content_dict: dict, config: dict):
    query = message_content_dict.get("query", "")
    sql = message_content_dict.get("sql", "")
    table_names = config.get("table_names")
    data_provider_id = config.get("data_provider_id")
    client_id = config.get("client_id")

    embedding_config = config.get("additional_instruction_embedding_api")
    success_history_config = config.get("success_history_vector_search")
    failed_history_config = config.get("failed_history_vector_search")

    with trace.get_tracer(__name__).start_as_current_span("Extract similar vectors") as span:
        span.set_attribute("table_names", table_names)

        if embedding_config and embedding_config.get("max_results") != 0:
            condition = f"(meta->>'table_name' in ({','.join(['\''+table_name+'\'' for table_name in table_names])}))"
            embedding_df, _ = search_by_natural_language(
                query=query,
                config=embedding_config,
                condition=condition,
                data_provider_id=data_provider_id
            )
        else:
            embedding_df = DataFrame()
        
        embedding_df["user_question"] = query
    
        table_names_check = " OR ".join([f"(meta->\'table_names\' @> \'[\"{table_name}\"]\'::jsonb)" for table_name in table_names])
        condition = f"(meta->>'sql' IS NOT NULL) AND (meta->>'table_names' IS NULL OR ({table_names_check})) AND (meta->>'rank' = '1' OR meta->>'rank' IS NULL)" 
    
        if success_history_config and success_history_config.get("max_results") != 0:
            success_history_df = search_for_similar_question_and_sql(
                config=success_history_config,
                query=query,
                sql=sql,
                condition=condition,
                data_provider_id=data_provider_id,
                client_id=client_id,
            )
        else:
            success_history_df = DataFrame()
        
        if failed_history_config and failed_history_config.get("max_results") != 0:
            failed_history_df = search_for_similar_question_and_sql(
                config=failed_history_config,
                query=query,
                sql=sql,
                condition=condition,
                data_provider_id=data_provider_id,
                client_id=client_id,
            )
        else:
            failed_history_df = DataFrame()
        
        return embedding_df, success_history_df, failed_history_df


def search_for_similar_question_and_sql(
    config: dict,
    query: str,
    sql: str,
    data_provider_id: int,
    client_id: int = None,
    condition: str = None,
):
    config["criteria"] = "distance < 0.5"

    embedding = get_embedding(query, config)
    df_history, _ = search_by_natural_language(
        query=query,
        config=config,
        embedding=embedding,
        data_provider_id=data_provider_id,
        client_id=client_id,
        condition=f"({condition} AND meta->>'embedded_field' = 'user_question')"
    )

    if sql:
        embedding = get_embedding(sql, config)

        df_history_sql, _ = search_by_natural_language(
            query=sql,
            config=config,
            embedding=embedding,
            data_provider_id=data_provider_id,
            client_id=client_id,
            condition=f"({condition} AND meta->>'embedded_field' = 'sql')"
        )

        df_history = concat([df_history, df_history_sql], axis=0)

    return df_history    


def map_column_names(columns_difinition: str, config: dict):
    columns = config.get("column_mapping") or {}
    for current_name, new_name in columns.items():
        columns_difinition = re.sub(
            rf"\b{current_name}\b", new_name, columns_difinition
        )
    return columns_difinition


def unmap_column_names(row: dict, config: dict):
    columns = config.get("column_mapping") or {}
    for new_name, current_name in columns.items():
        row["sql"] = re.sub(rf"\b{current_name}\b", new_name, row["sql"])
    return row


def set_prompt_columns(df: DataFrame = None, config: dict = None):
    response_column = config.get("response_column")

    if response_column not in df.columns:
        df["good_samples_prompt"] = ""
        df["bad_samples_prompt"] = ""
        df["past_failure_reasons"] = ""
    else:
        df = chunk_process(
            df=df,
            function=generate_sqlcoder_prompt_parameters,
            config=config,
            chunk_size=100,
            persist=False,
        )

    df.drop(
        columns=["distance", "score", "last_update_time"],
        inplace=True,
        errors="ignore",
    )
    return df


def construct_additional_instructions(embedding_df, config):
    embedding_df = embedding_df[["table_name", "column_name", "value"]].drop_duplicates()
    embedding_df["value"] = (
        embedding_df[["table_name", "column_name", "value"]]
        .groupby(["table_name", "column_name"])["value"]
        .transform(lambda x: ", ".join(f"'{v}'" for v in x))
    )
    embedding_df = embedding_df.drop_duplicates()
    embedding_df.loc[:, "column_name"] = embedding_df.loc[:, "column_name"].apply(
        map_column_names, config=config
    )
    return " \n\n".join(
        [
            f"* Column {row['column_name']} in table {row['table_name']} can have values {row['value']}"
            for _, row in embedding_df.iterrows()
        ]
    )


def vector_record_to_dict(vector_record: list):
    if not vector_record:
        return {}

    id, client_id, vector, meta = vector_record

    return dict(zip(
        ('id', 'client_id', 'vector', 'meta'), 
        (id, client_id, vector, meta)
    ))


def detect_create_table_statement(row: dict, table_names: list, config: dict):
    user_question = row["user_question"]
    user_question_embedding = get_embedding(user_question, config)
    text_search_fields = config.get("text_search_fields", [])
    if text_search_fields:
        text_query = " OR ".join([f"to_tsvector('english', meta->>'{field}')" for field in text_search_fields])
        text_condition = f"({text_query}) @@ to_tsquery('english', '{user_question}')"
    else:
        text_condition = "TRUE"

    condition = f"meta->>'table_name' in ({','.join(['\'' + table_name + '\'' for table_name in table_names])}) AND {text_condition}"

    if (data_provider_id := row.get("data_provider_id")) is None:
        raise ValueError("data_provider_id should not be empty")

    embedding_df, _ = search_by_natural_language(
        query=user_question,
        config=config,
        data_provider_id=data_provider_id,
        embedding=user_question_embedding,
        condition=condition
    )

    create_table_statement_list = []
    table_statements = ""

    df_with_table_columns = DataFrame(columns=["column_name", "table_name"])

    for table_name in table_names:
        df_with_table_name = embedding_df[embedding_df["table_name"] == table_name]
        if not df_with_table_name.empty:
            create_statement = construct_create_statement(df_with_table_name, table_name)
            create_table_statement_list.append(create_statement)
            df_with_table_columns = concat([df_with_table_columns, df_with_table_name[["column_name", "table_name"]]], ignore_index=True)
    if create_table_statement_list:
        table_statements = '\n\n'.join(create_table_statement_list)
    
    return table_statements, df_with_table_columns


def construct_create_statement(df: DataFrame, table_name: str):
    columns = []
    for _, row in df.iterrows():
        columns.append(f"\n {row['column_name']} {row['column_type']}")
    return f"CREATE TABLE {table_name} ({', '.join(columns)})"


def move_data_into_another_table(df: DataFrame, config: dict):
    delete_statement, returning_columns_str = construct_delete_command(df, config, config["remove_from_table"])
    command = f"""WITH deleted_rows AS ({delete_statement})
        INSERT INTO {config['add_to_table']} ({returning_columns_str})
        SELECT {returning_columns_str} FROM deleted_rows
    """
    execute_query(command)


def construct_delete_command(df: DataFrame, config: dict, table_name: str = None):
    pgvector_table_name = table_name or __get_pgvector_table_name(config)
    embedding_meta_column = config.get("embedding_meta_column") or "meta"
    delete_value_list = []
    columns = df.columns

    for _, row in df.iterrows():
        row_condition = []
        for column in columns:
            if column in ["id", "embedding"]:
                row_condition.append(f"{column} = {row[column]}")
            else:
                row_condition.append(f"{embedding_meta_column}->>'{column}' = '{row[column].replace("'", "''") if isinstance(row[column], str) else row[column]}'")
        delete_value = "(" + " AND ".join(row_condition) + ")"
        if delete_value is not None:
            delete_value_list.append(delete_value)

    delete_value_list_str = " OR ".join(delete_value_list)
    
    returning_columns_str = f"client_id, embedding, {embedding_meta_column}"
    
    command = f"""
        DELETE FROM {pgvector_table_name}
        WHERE {delete_value_list_str}
        RETURNING {returning_columns_str}
    """ if delete_value_list else None 

    return command, returning_columns_str


def get_data_from_table(
    config: dict,
    data_provider_id: int,
    condition=None,
    include_embedding=True,
    include_id=False
):
    include_row_keys = config.get("include_row_keys", [])
    pgvector_table_name = __get_pgvector_table_name(config)
    meta_column = config.get("meta_column") or "meta"

    if not pgvector_table_name:
            raise ValueError("pgvector_table_name is not found in config")

    use_distinct = config.get("use_distinct")

    select_keys_str = ", ".join(map(lambda key: f"{meta_column}->>'{key}' AS {key}", include_row_keys))
    select_keys_formatted_str = select_keys_str or "NULL as meta"

    if include_embedding:
        select_keys_formatted_str += ", embedding"
        include_row_keys.append("embedding")
    if include_id:
        select_keys_formatted_str += ", id"
        include_row_keys.append("id")

    where_condition = f"AND {condition}" if condition else ""

    command = f"""
        SELECT {'DISTINCT' if use_distinct else ''}
            {select_keys_formatted_str}
        FROM {pgvector_table_name}
        WHERE meta->>'data_provider_id' = '{data_provider_id}' {where_condition}
    """
    
    data , _ = execute_query(command)
    df = DataFrame(data, columns=include_row_keys)

    return df


def export_table_to_file(
    table_name: str, 
    output_path: str,
    config: dict = {},
):
    connection = get_connection()

    if not connection:
        logger.error("Failed to get a connection")
        return
    
    try:
        data, column_names = execute_query(f"""
            SELECT * 
            FROM {table_name} 
            {f"WHERE {config.get("condition")}" if config.get("condition") is not None else ''}
        """)

        df = DataFrame(data, columns=column_names)
        write_dataframe(df=df, output_path=output_path, convert_to_string=True)
        logger.info(f"Exported {table_name} to {output_path}")
    except Exception as e:
        logger.error(f"Error exporting table {table_name}: {e}")
    finally:
        connection_pool.putconn(connection)


def select_all_ask_intapp_user():
    internal_config = main_config.get("ask_intapp_users")

    select_command = f"""
        SELECT 
            id, 
            {internal_config.get('username_column_name')}, 
            {internal_config.get('password_column_name')}
        FROM {internal_config.get('pgvector_table_name')}
    """

    try:
        all_existing_users, _ = execute_query(select_command)
        return all_existing_users
    except Exception as e:
        logger.error(f"Error selecting user: {e}")
        return None


def select_ask_intapp_user(username: str, password: str):
    internal_config = main_config.get("ask_intapp_users")

    select_command = f"""
        SELECT id
        FROM {internal_config.get('pgvector_table_name')}
        WHERE 
            {internal_config.get('username_column_name')} = '{username}' 
            AND
            {internal_config.get('password_column_name')} = '{md5_encode(password)}'
    """

    try:
        existing_users, _ = execute_query(select_command)
        return existing_users[0] if len(existing_users) > 0 else None
    except Exception as e:
        logger.error(f"Error selecting user: {e}")
        return None


def insert_ask_intapp_user(username: str, password: str):
    internal_config = main_config.get("ask_intapp_users")

    insert_command = f"""
        INSERT INTO {internal_config.get('pgvector_table_name')} (
            {internal_config.get('username_column_name')},
            {internal_config.get('password_column_name')}
        ) 
        VALUES ('{username}', '{md5_encode(password)}')
        RETURNING id
    """

    select_command = f"""
        SELECT id
        FROM {internal_config.get('pgvector_table_name')}
        WHERE {internal_config.get('username_column_name')} = '{username}'
    """

    try:
        existing_users, _ = execute_query(select_command)
        if len(existing_users) > 0:
            raise Exception('User already exists.')
        execute_query(insert_command)
        return True
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return False


def md5_encode(text):
    if not text:
        raise Exception('`text` is required.')
    
    return md5(text.encode("utf-8")).hexdigest()


def select_ask_intapp_client(
    dealcloud_client_id: str, 
    product: str, 
    environment: str,
):
    internal_config = main_config.get("ask_intapp_clients")

    select_command = f"""
        SELECT client_id as id
        FROM {internal_config.get('pgvector_table_name')}
        WHERE 
            {internal_config.get('dealcloud_client_id_column_name')} = '{dealcloud_client_id}' 
            AND {internal_config.get('product_column_name')} = '{product}'
            AND {internal_config.get('environment_column_name')} = '{environment}'
    """

    try:
        clients, _ = execute_query(select_command)
        client = clients[0] if len(clients) > 0 else None
        return { "id": client[0] } if client else None
    except Exception as e:
        logger.error(f"Error selecting client: {e}")
        return None


def insert_ask_intapp_client(
    dealcloud_client_id: str, 
    product: str, 
    environment: str,
):
    internal_config = main_config.get("ask_intapp_clients")

    insert_command = f"""
        INSERT INTO {internal_config.get('pgvector_table_name')} (
            {internal_config.get('dealcloud_client_id_column_name')},
            {internal_config.get('product_column_name')},
            {internal_config.get('environment_column_name')}
        ) 
        VALUES ('{dealcloud_client_id}', '{product}', '{environment}')
        RETURNING client_id as id
    """

    try:
        inserted_clients, _ = execute_query(insert_command)
        client = inserted_clients[0] if len(inserted_clients) > 0 else None
        return { "id": client[0] } if client else None
    except Exception as e:
        logger.error(f"Error creating client: {e}")
        return None


def connection_check():
    try:
        res, _ = execute_query("SELECT 1")
        return res[0][0] == 1
    except Exception as e:
        logger.error(repr(e))
        return False
