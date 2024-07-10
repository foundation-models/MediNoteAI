from functools import cache
import json
import os
import re
from typing import Any

import psycopg2
import psycopg2.pool

from medinote import chunk_process, initialize, write_dataframe
from pandas import DataFrame, Series, concat, merge, read_parquet

from medinote.inference.inference_prompt_generator import row_infer

# Generatd with CHatGPT on 2021-08-25 15:00:00 https://chat.openai.com/share/133de26b-e5f5-4af8-a990-4a2b19d02254


main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or f"{os.path.dirname(__file__)}/../..",
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


def get_connection():

    try:
        # Get a connection from the pool
        connection = connection_pool.getconn()
        if connection:
            return connection
        else:
            # If no connection is available, create a new one
            connection = psycopg2.connect(
                user=pgvector_connection_config.get("user"),
                password=pgvector_connection_config.get("password"),
                host=pgvector_connection_config.get("host"),
                port=pgvector_connection_config.get("port"),
                database=pgvector_connection_config.get("database"),
            )
            return connection
    except Exception as e:
        print(f"Error getting connection: {e}")
        return None


def release_connection(connection):
    try:
        # Release the connection back to the pool
        connection_pool.putconn(connection)
    except Exception as e:
        print(f"Error releasing connection: {e}")


def close_connection(connection):
    try:
        # Close the connection
        connection.close()
    except Exception as e:
        print(f"Error closing connection: {e}")


def execute_query(query, params=None):
    connection = None
    cursor = None
    result = []
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(query, params)
        try:
            result = cursor.fetchall()
        except psycopg2.ProgrammingError as e:
            if "no results to fetch" in str(e):
                pass
            else:
                raise e
        connection.commit()
    except Exception as e:
        print(f"Error executing query: {e}")
        if connection:
            connection.rollback()
        # Recreate the connection if transaction fails
        if cursor:
            cursor.close()
        if connection:
            close_connection(connection)
        connection = get_connection()
    finally:
        if cursor:
            cursor.close()
        if connection:
            release_connection(connection)
    return result


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


def get_dataset_dict_and_df_2(config):
    dataset_parquet_path = config.get("dataset_parquet_path") if config else None
    if not dataset_parquet_path:
        raise ValueError(
            "You must provide either a dataset_dict or a dataset_parquet_file"
        )
    # dataset_df = read_parquet(dataset_parquet_path)
    # id_column = config.get(
    #     "id_column", "doc_id") if config else "doc_id"
    # content_column = config.get(
    #     "column2embed", "content") if config else "content"
    # dataset_dict = dataset_df.set_index(
    #     id_column)[content_column].to_dict()

    # return dataset_dict, dataset_df


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


def concat_near_vectors(row: dict, config: dict):
    if embedded_column_to_search := config.get("embedded_column_to_search"):

        df, _ = search_by_natural_language(
            query=row[embedded_column_to_search],
            config=config,
            embedding=row.get("embedding"),
        )
        similar_vector_prefix = config.get("similar_vector_prefix") or "source_"
        df = df.apply(
            lambda df_row: concat(
                [df_row, Series(row.rename(lambda x: f"{similar_vector_prefix}{x}"))]
            ),
            axis=1,
        )
        return df
    else:
        raise ValueError("embedded_column_to_search not found in config")


def search_by_natural_language(query: str, config: dict, embedding: list = None):

    if embedding is None:
        embedding = get_embedding(query, config)

    if len(embedding) == 0:
        raise ValueError("Invalid query vector")

    if not isinstance(embedding, list):
        embedding = embedding.tolist()

    df = search_by_vector(embedding, config=config)
    duplicate_column_to_check = config.get("duplicate_column_to_check")
    if duplicate_column_to_check:
        df.drop_duplicates(duplicate_column_to_check, inplace=True)
    df = df.sort_values(by="distance", ascending=True)
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


def search_by_vector(query_vector, config: dict):
    limit = config.get("limit", 10)
    database_name = config.get("vector_database_name", "pgvector")
    if database_name == "weaviate":
        collection_name = config.get("collection_name")
        client = get_weaviate_client(config=config)
        collection = client.collections.get(collection_name)

        from weaviate.classes.query import MetadataQuery

        documents = collection.query.near_vector(
            near_vector=list(query_vector),
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )
        data = []
        for obj in documents.objects:
            row = obj.properties
            row["distance"] = round(abs(obj.metadata.distance), 3)
            row["score"] = obj.metadata.score
            row["last_update_time"] = obj.metadata.last_update_time
            data.append(row)
        df = DataFrame(data)
    elif database_name == "pgvector":
        all_fields = list_all_fields(hashable_config=dict_to_hashable(config))
        include_row_keys = config.get("include_row_keys", [])
        include_row_keys = [field for field in all_fields if field in include_row_keys]

        pgvector_table_name = config.get("pgvector_table_name")
        embedding_column = config.get("embedding_column") or "embedding" or "embedding"
        if not pgvector_table_name:
            raise ValueError("pgvector_table_name not found in config")

        # conn = get_pgvector_connection(hashable_config=dict_to_hashable(config.get("pgvector_connection")))
        # cur = conn.cursor()
        data = execute_query(
            f"""
            SELECT {','.join(include_row_keys) + ',' if include_row_keys else ''} {embedding_column}, {embedding_column} <-> '{query_vector}' AS distance
            FROM {pgvector_table_name}
            ORDER BY distance
            LIMIT {limit}
        """
        )
        columns = include_row_keys.copy()
        columns.extend([embedding_column, "distance"])
        # data = cur.fetchall()

        df = DataFrame(data, columns=columns)
        df["distance"] = round(abs(df["distance"]), 3)
    if criteria := config.get("criteria"):
        df = df.query(criteria)
    return df


@cache
def list_all_fields(
    table_name: str = None, table_schema: str = None, hashable_config: tuple = None
):
    config = hashable_to_dict(hashable_config) if hashable_config else {}
    table_name = table_name or config.get("pgvector_table_name")
    table_schema = table_schema or config.get("pgvector_table_schema") or "public"
    all_fields = execute_query(
        f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        AND table_schema = '{table_schema}'
    """
    )
    return [field[0] for field in all_fields]


def search_by_id(
    id: str = None,
    query_vector: list = None,
    vector_database_name: str = "weaviate",
    client: Any = None,
    embedding_column: str = "embedding",
    limit: int = 2,
    config: dict = None,
):
    """
    Search for similar documents based on the given ID and query vector.

    Args:
        id (str): The ID of the document to search for.
        query_vector (list): The query vector used for similarity search.
        vector_database_name (str): The name of the vector database to search in.
        client (weaviate.client): The Weaviate client instance. If not provided, a new client will be created.
        embedding_column (str): The name of the column containing the embeddings in the dataset.
        limit (int): The maximum number of similar documents to return.

    Returns:
        DataFrame: A pandas DataFrame containing the similar documents and their metadata.
    """
    id_column = config.get("id_column", "doc_id") if config else "doc_id"
    dataset_dict, dataset_df = get_dataset_dict_and_df(config)
    text = dataset_dict.get(id)
    query_vectors = dataset_df[dataset_df[id_column] == id][embedding_column].tolist()
    source_doc_ids = dataset_df[dataset_df[id_column] == id]["doc_id"].tolist()
    collection_name = config.get("collection_name")
    client = client or get_weaviate_client()
    collection = client.collections.get(collection_name)
    data = []

    from weaviate.classes.query import MetadataQuery

    for query_vector, source_doc_id in zip(query_vectors, source_doc_ids):
        documents = collection.query.near_vector(
            near_vector=list(query_vector),
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )
        for obj in documents.objects:
            row = obj.properties
            if row.get("doc_id") == source_doc_id:
                # if the source document is returned as a similar document, skip it
                continue
            row["source_doc_id"] = source_doc_id
            # here id is the file_path of the source document
            row["source_ref"] = id
            # finding path of the near document
            row["near_ref"] = dataset_df[dataset_df["doc_id"] == row["doc_id"]][
                "file_path"
            ].values[0]
            # we mark cases that we have found the chunks of the same document
            if row["source_ref"] == row["near_ref"]:
                row["source_distance"] = 0
            row["distance"] = round(abs(obj.metadata.distance), 3)
            row["score"] = obj.metadata.score
            row["last_update_time"] = obj.metadata.last_update_time
            data.append(row)
    return DataFrame(data)


def cross_search_all_docs(
    exclude_ids=None,
    config: dict = None,
    persist: bool = True,
):
    """
    Cross searches all documents in the dataset, excluding the specified IDs.

    Args:
        exclude_ids (list): List of document IDs to exclude from the search. Default is an empty list.

    Returns:
        DataFrame: A DataFrame containing the search results.
    """
    logger.info("Cross searching all documents")
    df = DataFrame()
    dataset_dict, _ = get_dataset_dict_and_df(config)
    client = get_weaviate_client(config=config)

    for id in dataset_dict.keys():
        if exclude_ids is None or id not in exclude_ids:
            df = concat(
                [df, search_by_id(id, limit=10, client=client, config=config)], axis=0
            )
    client.close()
    cross_distance_output_path = config.get("cross_distance_output_path")
    if persist and cross_distance_output_path:
        df = df.astype(str)
        write_dataframe(df=df, output_path=cross_distance_output_path)
    return df


def extract_data_objectst(
    row: dict,
    column2embed: str,
    embedding_column: str = "embedding",
    include_row_keys: list = None,
):
    if column2embed not in row:
        raise ValueError(f"embedding_column: {column2embed} not found in row")
    try:
        properties = {column2embed: row[column2embed]}
        embedding = row[embedding_column]
        if include_row_keys:
            for key in include_row_keys:
                properties[key] = row.get(key)
        return DataObject(
            properties=properties,
            vector=embedding.tolist(),
        )
    except Exception as e:
        logger.error(f"Error embedding row: {repr(e)}")
        raise e


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


def get_weaviate_client(config: dict = None):

    import weaviate

    weaviate_host = config.get("weaviate_host")
    weaviate_grpc = config.get("weaviate_grpc")
    client = weaviate.connect_to_custom(
        http_host=weaviate_host,
        http_port=8080,
        http_secure=False,
        grpc_host=weaviate_grpc,
        grpc_port=50051,
        grpc_secure=False,
    )  # Connect to Weaviate
    return client


def create_or_update_weaviate_vdb_collection(
    df: DataFrame = None,
    config: dict = None,
    column2embed: str = None,
    embedding_column: str = "embedding",
    recreate: bool = True,
):
    """
    Creates collections in Weaviate Vector Database (VDB) and inserts data objects into the collections.

    Args:
        df (DataFrame, optional): The input DataFrame containing the data to be inserted into the collections. If not provided, the data will be read from the output_path specified in the configuration.
        text_field (str, optional): The name of the field in the DataFrame that contains the text data. If not provided, the default text_field specified in the configuration will be used.
        column2embed (str, optional): The name of the field in the DataFrame that contains the embeddings. If not provided, the default column2embed specified in the configuration will be used.
        embedding_column (str, optional): The name of the column in the DataFrame that contains the data to be embedded. If not provided, the default embedding_column specified in the configuration will be used.
        recreate (bool, optional): If True, recreates the collection if it already exists. If False, retrieves the existing collection. Defaults to True.
    """
    # Code to create NOW and FUTURE collections
    # WeaviateVectorClient stores text in this field by default
    # WeaviateVectorClient stores embeddings in this field by default
    if config is None:
        raise "config is None"
    if df is None:
        output_path = config.get("output_path")
        logger.debug(f"Reading the input parquet file from {output_path}")
        df = read_parquet(output_path)

    # http weaviate_url for your cluster (weaviate required for vector index usage)
    client = get_weaviate_client(config=config)

    collection_name = config.get("collection_name")
    # index to demonstrate the VectorStore impl
    try:
        collection = create_collection(client, collection_name)
    except Exception:
        if recreate:
            client.collections.delete(collection_name)
            collection = create_collection(client, collection_name)
        else:
            collection = client.collections.get(collection_name)

    column2embed = column2embed or config.get(
        "column2embed"
    )  # TODO: Check if None is valid value
    embedding_column = config.get("embedding_column") or "embedding"

    include_row_keys = config.get("include_row_keys") or config.get("selected_columns")

    # load some sample data
    data_objects = (
        df.apply(
            extract_data_objectst,
            axis=1,
            column2embed=column2embed,
            embedding_column=embedding_column,
            include_row_keys=include_row_keys,
        ).tolist()
        if os.getenv("USE_DASK", "False") == "True"
        else df.apply(
            extract_data_objectst,
            axis=1,
            column2embed=column2embed,
            embedding_column=embedding_column,
            include_row_keys=include_row_keys,
        ).tolist()
    )
    collection.data.insert_many(data_objects)
    return df


def create_collection(client, collection_name):
    return client.collections.create(
        collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE  # select prefered distance metric
        ),
    )


@cache
def get_pgvector_connection(hashable_config: tuple = None):
    import psycopg2
    from dotenv import load_dotenv

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


def insert_into_vector_database(df, config):
    if config.get("recreate"):
        create_pgvector_table(config=config)
    
    embedding_instruct = config.get("embedding_instruct")
    config["embedding_instruct"] = None
    df['embedding'] = df["sql"].apply(lambda x: get_embedding(x, config))
    df['embedded_field'] = 'sql' 
    command = construct_insert_command(df=df, config=config)
    execute_query(command)
    df['embedding'] = df["user_question"].apply(lambda x: get_embedding(x, config))
    df['embedded_field'] = 'user_question' 
    command = construct_insert_command(df=df, config=config)
    execute_query(command)
    config["embedding_instruct"] = embedding_instruct

def create_pgvector_table(
    config: dict,
):
    embedding_column = config.get("embedding_column") or "embedding"
    vector_dimension = config.get("vector_dimension") or 1024
    include_row_keys = (
        config.get("include_row_keys") or config.get("selected_columns") or []
    )
    sql_statements = [f"{x} TEXT" for x in include_row_keys]
    pgvector_table_name = config.get("pgvector_table_name")

    conn = get_pgvector_connection(
        hashable_config=dict_to_hashable(main_config.get("pgvector_connection"))
    )
    cur = conn.cursor()

    cur.execute(
        f"""
        DROP TABLE IF EXISTS {pgvector_table_name}
    """
    )
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {pgvector_table_name} (
            id SERIAL PRIMARY KEY,              
            {embedding_column} VECTOR({vector_dimension}),
            {','.join(sql_statements)}              
        )
    """
    )
    conn.commit()


# def execute_insert_into_pgvector_table(
#     row: dict,
#     pgvector_table_name: str,
#     embedding_column: str,
#     include_row_keys: list,
#     config: dict = None,
# ):
#     # conn = get_pgvector_connection(
#     #     hashable_config=dict_to_hashable(config.get("pgvector_connection"))
#     # )
#     # cur = conn.cursor()
#     try:
#         command = construct_insert_command(
#             [row], pgvector_table_name, embedding_column, include_row_keys
#         )
#         execute_query(command)
#         # cur.execute(command)
#         # conn.commit()
#     except Exception as e:
#         logger.error(f"Error inserting row: {repr(e)}")
#         raise e


# def execute_insert_dataframe_into_pgvector_table(
#     df: DataFrame,
#     pgvector_table_name: str,
#     embedding_column: str,
#     include_row_keys: list,
#     config: dict = None,
#     conn: object = None,
# ):
#     conn = conn or get_pgvector_connection(
#         hashable_config=dict_to_hashable(config.get("pgvector_connection"))
#     )
#     cur = conn.cursor()
#     try:
#         rows = [row for _, row in df.iterrows()]
#         command = construct_insert_command(
#             rows, pgvector_table_name, embedding_column, include_row_keys
#         )

#         cur.execute(command)
#         conn.commit()

#     except Exception as e:
#         logger.error(f"Error inserting row: {repr(e)}")
#         raise e


def construct_insert_command(df: DataFrame, config: dict):
    embedding_column = config.get("embedding_column") or "embedding"
    include_row_keys = config.get("include_row_keys")
    insert_value_list = []
    for index, row in df.iterrows():
        embedding_value = row.get(embedding_column)
        insert_value = (
            [embedding_value]
            if isinstance(embedding_value, list)
            else [embedding_value.tolist()]
        )
        for key in include_row_keys:
            if key:
                value = row.get(key)
                if isinstance(row.get(key), str):
                    value = value.replace("'", "''")
                insert_value.append(value)
            else:
                include_row_keys.pop(include_row_keys.index(key))
        insert_value = ", ".join(f"'{value}'" for value in insert_value)
        insert_value_list.append(f"({insert_value})")

    insert_value_str = ", ".join(insert_value_list)

    pgvector_table_name = config.get("pgvector_table_name")

    command = f"""
        INSERT INTO {pgvector_table_name} ({embedding_column}, {','.join(include_row_keys)}) 
        VALUES {insert_value_str}
    """

    return command


def set_additional_instructions(row: dict, config: dict):
    user_question = row.get("user_question")
    embedding_df, _ = search_by_natural_language(
        query=user_question,
        config=config.get("additional_instruction_embedding_api"),
        embedding=row.get("embedding"),
    )
    embedding_df["user_question"] = user_question
    row["additional_instructions"] = construct_additional_instructions(
        embedding_df, config
    )

    return row


def generate_sqlcoder_prompt_parameters(message_content_dict: dict, config: dict):
    message_history = message_content_dict.get("message_history", [])
    original_query = message_content_dict.get("query", "")
    incorrect_sql = message_content_dict.get("sql", "")
    user_question = message_content_dict.get("user_question", "")
    result_str = message_content_dict.get("result", "{}")
    result = json.loads(result_str)
    reason = result.get("error", "") if result else ""

    query = user_question or original_query

    embedding_df, embedding = search_by_natural_language(
        query=query,
        config=config.get("additional_instruction_embedding_api"),
    )
    embedding_df["user_question"] = query
    row = {}
    row["additional_instructions"] = construct_additional_instructions(
        embedding_df, config
    )

    df_history = search_for_similar_question_and_sql(
        config=config.get("success_history_vector_search"),
        query=query,
        sql=incorrect_sql,
        embedding=embedding,
    )

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
    row["user_question"] = original_query
    row["past_failure_reasons"] = reason if reason else ""
    row["create_table_statements"] = config.get(
        "create_table_statements"
    ) or message_content_dict.get("create_table_statements", "")
    row["insert_statements"] = (
        message_content_dict.get("insert_statements")
        or config.get("insert_statements")
        or ""
    )
    return row


def search_for_similar_question_and_sql(config, query, sql, embedding):
    config["criteria"] = f"embedded_field == 'user_question' and distance < 0.5"

    df_history, _ = search_by_natural_language(
        query=query,
        config=config,
        embedding=embedding,
    )

    if sql:
        config["criteria"] = f"embedded_field == 'sql' and distance < 0.5"
        embedding = get_embedding(sql, config)

        df_history_sql, _ = search_by_natural_language(
            query=sql,
            config=config,
            embedding=embedding,
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


def handle_table_statements(row: dict, config: dict = None):
    if row.get("table_names"):
        table_statement = create_table_statements(row.get("table_names").split(","))
    else:
        if config.get("table_statement"):
            table_statement = config.get("table_statement")
        else:
            table_statement = create_table_statements(
                config.get("table_names"),
                config.get("generic_table_name"),
                config.get("data_provider_id"),
                config.get("schema_name"),
            )

        row["table_names"] = ", ".join(config.get("table_names"))

    row["create_table_statements"] = map_column_names(table_statement, config)

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
    embedding_df = embedding_df[["column_name", "value"]].drop_duplicates()
    embedding_df["value"] = (
        embedding_df[["column_name", "value"]]
        .groupby("column_name")["value"]
        .transform(lambda x: ", ".join(x))
    )
    embedding_df = embedding_df[["column_name", "value"]].drop_duplicates()
    embedding_df["column_name"] = embedding_df["column_name"].apply(
        map_column_names, config=config
    )
    return " \n\n".join(
        [
            f"* Column {row['column_name']} can have values {row['value']}"
            for _, row in embedding_df.iterrows()
        ]
    )
