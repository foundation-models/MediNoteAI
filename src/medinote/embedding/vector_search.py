import ast
import hashlib
import os

from medinote import initialize
from medinote import write_dataframe
from pandas import Series, concat, merge, read_parquet

from medinote.inference.inference_prompt_generator import row_infer

# Generatd with CHatGPT on 2021-08-25 15:00:00 https://chat.openai.com/share/133de26b-e5f5-4af8-a990-4a2b19d02254

try:
    from llama_index import Document
except ImportError:
    from llama_index.core import Document

from pandas import DataFrame, read_parquet
import weaviate
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure, VectorDistances


main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=f"{os.path.dirname(__file__)}/../..",
)


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
    if embedded_column_to_search:= config.get("embedded_column_to_search"):    
                
        df = search_by_natural_language(query=row[embedded_column_to_search], config=config, embedding = row.get("embedding"))
        similar_vector_prefix = config.get("similar_vector_prefix") or "source_"
        df = df.apply(lambda df_row: concat([df_row, Series(row.rename(lambda x: f"{similar_vector_prefix}{x}"))]), axis=1)
        return df
    else:
        raise ValueError("embedded_column_to_search not found in config")

def search_by_natural_language(query: str, config: dict, embedding: list = None):
    row = {"query": query}
    if embedding is None:
        row = row_infer(row=row, config=config)
        embedding = row.get("embedding")
    
    if len(embedding) == 0:
        raise ValueError("Invalid query vector")

    if not isinstance(embedding, list):
        embedding = embedding.tolist()
    
    df = search_by_vector(embedding, config=config)
    duplicate_column_to_check = config.get("duplicate_column_to_check")
    if duplicate_column_to_check:
        df.drop_duplicates(duplicate_column_to_check, inplace=True)
    df = df.sort_values(by="distance", ascending=False)
    max_results = config.get("max_results")
    if max_results:
        df = df[:max_results]
    return df


def search_by_vector(query_vector, config: dict):
    collection_name = config.get("collection_name")
    limit = config.get("limit", 10)
    database_name = config.get("vector_database_name", "weaviate")
    if database_name == "weaviate":
        client = get_weaviate_client(config=config)
        collection = client.collections.get(collection_name)

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
    elif database_name == "pgvector":
        conn = get_pgvector_connection()
        cur = conn.cursor()
        include_row_keys = config.get("include_row_keys", ["id"])
        pgvector_tablename = config.get("pgvector_tablename")
        if not pgvector_tablename:
            raise ValueError("pgvector_tablename not found in config")
        cur.execute(f"""
            SELECT {','.join(include_row_keys)}, embedding, embedding <-> '{query_vector}' AS distance
            FROM {pgvector_tablename}
            ORDER BY distance
            LIMIT {limit}
        """)
        data = cur.fetchall()
    return DataFrame(data)


def search_by_id(
    id: str = None,
    query_vector: list = None,
    vector_database_name: str = "weaviate",
    client: weaviate.client = None,
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
        raise ValueError(
            f"embedding_column: {column2embed} not found in row"
        )
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

    column2embed = column2embed or config.get("column2embed")  # TODO: Check if None is valid value
    if config.get("embedding_column"):
        embedding_column = config.get("embedding_column")

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

def get_pgvector_connection(config: dict = None):
    import psycopg2
    from dotenv import load_dotenv

    load_dotenv()
    db_config = {
        "dbname": config.get("pgvector_database_name") or os.getenv("DB_NAME"),
        "user":  config.get("pgvector_user") or os.getenv("DB_USER"),
        "password":  config.get("pgvector_password") or os.getenv("DB_PASSWORD"),
        "host":  config.get("pgvector_host") or os.getenv("DB_HOST") or "localhost",
        "port":  config.get("pgvector_port") or os.getenv("DB_PORT") or 6432,
    }
    conn = psycopg2.connect(**db_config)
    return conn

def create_or_update_pgvector_table(
    df: DataFrame = None,
    config: dict = None,
    column2embed: str = None,
    embedding_column: str = None,
    recreate: bool = True,
):
    if config is None:
        raise "config is None"
    if df is None:
        output_path = config.get("output_path")
        logger.debug(f"Reading the input parquet file from {output_path}")
        df = read_parquet(output_path)

    column2embed = column2embed or config.get("column2embed")
    embedding_column = embedding_column or config.get("embedding_column") or "embedding"
    vector_dimension = config.get("vector_dimension") or 1024
    include_row_keys = config.get("include_row_keys") or config.get("selected_columns") or []
    sql_statements = [f"{x} TEXT" for x in include_row_keys]
    pgvector_table_name = config.get("pgvector_table_name")

    if recreate:
        conn = get_pgvector_connection(config=config)
        cur = conn.cursor()

        cur.execute(f"""
            DROP TABLE IF EXISTS {pgvector_table_name}
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {pgvector_table_name} (
                id SERIAL PRIMARY KEY,              
                {embedding_column} VECTOR({vector_dimension}),
                {','.join(sql_statements)}              
            )
        """)
        conn.commit()

    # df.apply(
    #     execute_insert_into_pgvector_table,
    #     axis=1,
    #     pgvector_table_name=pgvector_table_name,
    #     embedding_column=embedding_column,
    #     include_row_keys=include_row_keys,
    #     config=config
    # )

    execute_insert_dataframe_into_pgvector_table(df,
        pgvector_table_name,
        embedding_column,
        include_row_keys,
        config
    )

    return df

def execute_insert_into_pgvector_table(row: dict,
                                       pgvector_table_name: str,
                                       embedding_column: str,
                                       include_row_keys: list,
                                       config: dict = None):
    conn = get_pgvector_connection(config=config)
    cur = conn.cursor()
    try:
        insert_values = [f'{row.get(embedding_column).tolist()}']
        for key in include_row_keys:
            value = row.get(key)
            value = value.replace("'", "''")
            insert_values.append(value)

        insert_values = ', '.join(f"'{value}'" for value in insert_values)

        command = f"""
            INSERT INTO {pgvector_table_name} ({embedding_column}, {','.join(include_row_keys)}) 
            VALUES ({insert_values})
        """

        cur.execute(command)
        conn.commit()
    except Exception as e:
        logger.error(f"Error inserting row: {repr(e)}")
        raise e


def execute_insert_dataframe_into_pgvector_table(df: DataFrame,
                                       pgvector_table_name: str,
                                       embedding_column: str,
                                       include_row_keys: list,
                                       config: dict = None):
    conn = get_pgvector_connection(config=config)
    cur = conn.cursor()
    try:
        insert_value_list = []
        for _, row in df.iterrows():
            insert_value = [f'{row.get(embedding_column).tolist()}']
            for key in include_row_keys:
                value = row.get(key)
                if isinstance(row.get(key), str):
                    value = value.replace("'", "''")
                insert_value.append(value)
            insert_value = ', '.join(f"'{value}'" for value in insert_value)
            insert_value_list.append(f'({insert_value})')

        insert_value_str = ', '.join(insert_value_list)

        command = f"""
            INSERT INTO {pgvector_table_name} ({embedding_column}, {','.join(include_row_keys)}) 
            VALUES {insert_value_str}
        """

        cur.execute(command)
        conn.commit()

    except Exception as e:
        logger.error(f"Error inserting row: {repr(e)}")
        raise e