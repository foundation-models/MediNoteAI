import os

from medinote import initialize
from medinote import write_dataframe
from medinote.curation.rest_clients import generate_via_rest_client
from pandas import Series, concat, merge, read_parquet

# Generatd with CHatGPT on 2021-08-25 15:00:00 https://chat.openai.com/share/133de26b-e5f5-4af8-a990-4a2b19d02254
from medinote.embedding.embedding_generation import retrieve_embedding

try:
    from llama_index import Document, VectorStoreIndex
except ImportError:
    from llama_index.core import Document, VectorStoreIndex

from pandas import DataFrame, read_parquet
import weaviate
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure, VectorDistances

try:
    from llama_index.storage import StorageContext
    from llama_index.vector_stores.opensearch import (
        OpensearchVectorClient,
        OpensearchVectorStore,
    )
except ImportError as e:
    print(f"ignoring error {e}")
    


_, logger = initialize()


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


def opensearch_vector_query_for_dataframe(
    row: Series,
    input_column: str,
    output_column: str,
    dataset_dict: dict = None,
    text_field: str = None,
    column2embed: str = None,
    vector_store: OpensearchVectorStore = None,
    return_dataset: bool = False,
    return_content: bool = False,
    return_doc_id: bool = False,
):
    row[output_column] = opensearch_vector_query(
        row[input_column],
        dataset_dict=dataset_dict,
        text_field=text_field,
        column2embed=column2embed,
        vector_store=vector_store,
        return_dataset=return_dataset,
        return_content=return_content,
        return_doc_id=return_doc_id,
    )
    return row


def get_dataset_dict_and_df_2(config):
    dataset_parquet_path = (
        config.get("dataset_parquet_path") if config else None
    )
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
    dataset_parquet_path = (
        config.get("dataset_parquet_path") if config else None
    )
    if not dataset_parquet_path:
        raise ValueError(
            "You must provide either a dataset_dict or a dataset_parquet_file"
        )
    dataset_df = read_parquet(dataset_parquet_path)
    id_column = (
        config.get("id_column", "doc_id") if config else "doc_id"
    )
    content_column = (
        config.get("embedding_column", "content") if config else "content"
    )
    dataset_dict = dataset_df.set_index(id_column)[content_column].to_dict()

    return dataset_dict, dataset_df


def opensearch_vector_query(
    query: str,
    query_vector: list = None,
    text_field: str = None,
    column2embed: str = None,
    id_field: str = None,
    vector_store: OpensearchVectorStore = None,
    return_dataset: bool = False,
    return_content: bool = False,
    return_doc_id: bool = False,
    config: dict = None,
):
    if not vector_store:
        vector_store = get_vector_store(text_field, column2embed)

    if query_vector is None:
        query = [query] if isinstance(query, str) else query
        logger.debug(f"query: {query}")

        # OpensearchVectorClient encapsulates logic for a
        # single opensearch index with vector search enabled

        payload = {"input": [query]}
        embedding_url = config.get("embedding_url")
        id_field = (
            id_field or config.get("id_column", "doc_id")
            if config
            else "doc_id"
        )
        embeddings = generate_via_rest_client(
            payload=payload, inference_url=embedding_url
        )
        if isinstance(embeddings, str) and '{"error"' in embeddings:
            raise Exception("Embedding service is not available")
        query_vector = embeddings[0]

    similarity_top_k = (
        config.get("similarity_top_k", 10) if config else 10
    )

    vector_store_query = VectorStoreQuery(
        query_vector=query_vector, similarity_top_k=similarity_top_k
    )

    nodes = vector_store.query(vector_store_query).nodes
    dataset_dict, dataset_df = get_dataset_dict_and_df(config)

    if return_dataset and dataset_df:
        return dataset_df[
            dataset_df[id_field].isin([node.ref_doc_id for node in nodes])
        ]
    elif return_content and dataset_dict:
        return [dataset_dict.get(node.ref_doc_id) for node in nodes]
    elif return_doc_id:
        return [node.ref_doc_id for node in nodes]

    documents = []
    for node in nodes:
        # if hit.distance >= target_distance:

        page = dataset_dict.get(node.ref_doc_id)
        document = Document(
            doc_id=node.ref_doc_id,
            text=page if type(page) is str else page.text,
        )
        documents.append(document)

    # return documents[:return_limit]
    return documents

def search_by_natural_language(query: str, config: dict):
    _, query_vector = retrieve_embedding(query=query, config=config)
    
    if not isinstance(query_vector, list) or len(query_vector) == 0:
        raise ValueError("Invalid query vector")

    df = search_by_vector(query_vector, config=config)
    duplicate_column_to_check = config.get("duplicate_column_to_check")
    if duplicate_column_to_check:  
        df.drop_duplicates(duplicate_column_to_check, inplace=True) 
    df = df.sort_values(by='distance', ascending=False)
    max_results = config.get("max_results")
    if max_results:
        df = df[:max_results]
    return df

def search_by_vector(query_vector, config: dict):
    collection_name = config.get("collection_name")
    limit = config.get("limit", 10)
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
    id_column = (
        config.get("id_column", "doc_id") if config else "doc_id"
    )
    dataset_dict, dataset_df = get_dataset_dict_and_df(config)
    text = dataset_dict.get(id)
    query_vectors = dataset_df[dataset_df[id_column] == id][embedding_column].tolist()
    source_doc_ids = dataset_df[dataset_df[id_column] == id]["doc_id"].tolist()
    if vector_database_name == "opensearch":
        documents = opensearch_vector_query(query=text, query_vector=query_vector)
        return documents
    else:
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
                    continue
                row["source_doc_id"] = source_doc_id
                row["source_ref"] = id
                row["near_ref"] = dataset_df[dataset_df["doc_id"] == row["doc_id"]][
                    "file_path"
                ].values[0]
                if row["source_ref"] == row["near_ref"]:
                    row["source_distance"] = 0
                row["distance"] = round(abs(obj.metadata.distance), 3)
                row["score"] = obj.metadata.score
                row["last_update_time"] = obj.metadata.last_update_time
                data.append(row)
        return DataFrame(data)


# def search_document_by_file_name(file_name: str = None,
#                                  file_name_column: str = 'file_name',
#                                  ):

#     doc_ids = dataset_df[dataset_df[file_name_column]
#                          == file_name]['doc_id', 'embedding'].tolist()


#     df = DataFrame()
#     for doc_id in doc_ids:
#         df = concat([df, search_by_id(doc_id)], axis=0)
#     cross_distance_output_path = config.get("cross_distance_output_path")
#     if cross_distance_output_path:
#         df.to_parquet(cross_distance_output_path)
#     return df


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
    cross_distance_output_path = config.get(
        "cross_distance_output_path"
    )
    if persist and cross_distance_output_path:
        df = df.astype(str)
        write_dataframe(df=df, output_path=cross_distance_output_path)
    return df


def get_collection_name(config: dict = None):
    collection_name = config.get("collection_name")
    # chunk_size = config.get("pdf_reader").get("chunk_size")
    # chunk_overlap = config.get("pdf_reader").get("chunk_overlap")
    # if chunk_size and chunk_overlap:
    #     collection_name = f"{collection_name}_{chunk_size}_{chunk_overlap}_{int(time())}"
    return collection_name


def get_vector_store(
    text_field: str = None,
    column2embed: str = None,
    config: dict = None,
):

    opensearch_url = config.get("opensearch_url")
    text_field = (
        text_field or config.get("text_field", "content")
        if config
        else "content"
    )
    column2embed = (
        column2embed or config.get("column2embed", "embedding")
        if config
        else "embedding"
    )
    vector_dimension = config.get("vector_dimension")
    collection_name = config.get("collection_name")

    logger.debug(
        f"collection_name: {collection_name} opensearch_url: {opensearch_url} text_field: {text_field} column2embed: {column2embed}"
    )
    client = OpensearchVectorClient(
        endpoint=opensearch_url,
        index=collection_name,
        dim=vector_dimension,
        column2embed=column2embed,
        text_field=text_field,
    )
    vector_store = OpensearchVectorStore(client)
    return vector_store


def add_similar_documents(
    df: DataFrame = None,
    text_field: str = None,
    column2embed: str = None,
    persist: bool = True,
    config: dict = None,
):
    vector_store = get_vector_store(text_field, column2embed)
    output_path = config.get("output_path")
    content_column = (
        config.get("column2embed", "content") if config else "content"
    )
    if df in None:
        input_path = config.get("input_path")
        if not input_path:
            raise ValueError(f"No input_path found.")
        logger.debug(f"Reading the input parquet file from {input_path}")
        df = read_parquet(input_path)

    dataset_dict, _ = get_dataset_dict_and_df(config)
    df = (
        df.apply(
            opensearch_vector_query_for_dataframe,
            axis=1,
            input_column=content_column,
            output_column="similar_doc_id_list",
            #   vector_store=vector_store,
            dataset_dict=dataset_dict,
            return_doc_id=True,
            vector_store=vector_store,
        )
        if os.getenv("USE_DASK", "False") == "True"
        else df.parallel_apply(
            opensearch_vector_query_for_dataframe,
            axis=1,
            input_column=content_column,
            output_column="similar_doc_id_list",
            #   vector_store=vector_store,
            dataset_dict=dataset_dict,
            return_doc_id=True,
            vector_store=vector_store,
        )
    )
    if persist and output_path:
        logger.debug(f"Saving the embeddings to {output_path}")
        write_dataframe(df=df, output_path=output_path)
    return df


def extract_data_objectst(
    row: dict,
    column2embed: str,
    index_column: str = "doc_id",
    embedding_column: str = "embedding",
    include_row_keys: list = None,
):
    if embedding_column not in row or index_column not in row or column2embed not in row: 
        raise ValueError(
            f"embedding_column: {embedding_column} or index_column: {index_column} not found in row"
        )
    try:
        text=row.get(column2embed),
        #         text = text[0] if isinstance(text, (list, tuple)) else text
        doc_id=row.get(index_column) # or hashlib.sha256(text.encode()).hexdigest()
        properties={column2embed: text, index_column: doc_id}
        if include_row_keys:
            for key in include_row_keys:
                properties[key] = row.get(key)
        return DataObject(
            properties=properties,
            vector=list(row[embedding_column]),
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
    index_column: str = None,
    embedding_column: str = None,
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

    collection_name = get_collection_name(config=config)
    # index to demonstrate the VectorStore impl
    # collection_name = get_collection_name()
    try:
        collection = create_collection(client, collection_name)
    except Exception:
        if recreate:
            collection = client.collections.delete(collection_name)
            collection = create_collection(client, collection_name)
        else:
            collection = client.collections.get(collection_name)

    column2embed = column2embed or config.get("column2embed")
    embedding_column = embedding_column or config.get("embedding_column")
    index_column = index_column or config.get("index_column")
    include_row_keys = config.get("include_row_keys") or config.get("selected_columns")

    # load some sample data
    data_objects = (
        df.apply(
            extract_data_objectst,
            axis=1,
            column2embed=column2embed,
            index_column=index_column,
            embedding_column=embedding_column,
            include_row_keys=include_row_keys,
        ).tolist()
        if os.getenv("USE_DASK", "False") == "True"
        else df.parallel_apply(
            extract_data_objectst,
            axis=1,
            column2embed=column2embed,
            index_column=index_column,
            embedding_column=embedding_column,
            include_row_keys=include_row_keys,
        ).tolist()
    )
    collection.data.insert_many(data_objects)

def create_open_search_vdb_collections(
    df: DataFrame = None,
    config: dict = None,        
    text_field: str = None,
    column2embed: str = None,
    embedding_column: str = None,
):
    # Code to create NOW and FUTURE collections
    # OpensearchVectorClient stores text in this field by default
    # OpensearchVectorClient stores embeddings in this field by default
    config = config or embedding_conf

    if df is None:
        output_path = config.get("output_path")
        logger.debug(f"Reading the input parquet file from {output_path}")
        df = read_parquet(output_path)

    # http opensearch_url for your cluster (opensearch required for vector index usage)
    opensearch_url = config.get("opensearch_url")
    # index to demonstrate the VectorStore impl
    # collection_name = get_collection_name()
    vector_dimension = config.get("vector_dimnesion")
    text_field = text_field or config.get("text_field")
    column2embed = column2embed or config.get("column2embed")
    embedding_columnlumn = embedding_column or config.get("embedding_column")
    chunk_size = config.get("chunk_size")

    # load some sample data
    documents = (
        df.apply(
            extract_document,
            axis=1,
            column_name=column2embed,
            column2embed=column2embed,
        ).tolist()
        if os.getenv("USE_DASK", "False") == "True"
        else df.parallel_apply(
            extract_document,
            axis=1,
            column_name=column2embed,
            column2embed=column2embed,
        ).tolist()
    )
    collection_name = get_collection_name(config=config)

    if len(documents) > 0:
        # OpensearchVectorClient encapsulates logic for a
        # single opensearch index with vector search enabled
        client = OpensearchVectorClient(
            endpoint=opensearch_url,
            index=collection_name,
            dim=vector_dimension,
            column2embed=column2embed,
            text_field=text_field,
        )
        # initialize vector store
        vector_store = OpensearchVectorStore(client)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # initialize an index using our sample data and the client we just created
        document_chunks = (
            chunk_documents(documents, chunk_size) if chunk_size else [documents]
        )

        indexes = []
        for chunk in document_chunks:
            index = VectorStoreIndex.from_documents(
                documents=chunk, storage_context=storage_context
            )
            indexes.append(index)
        return indexes[-1]
        # index.set_index_id(f"{knowledge.id}")
        # index.storage_context.persist(persist_dir=f"{VDB_ROOT}/{knowledge.id}", vector_store_fname=vector_store_fname)




def create_collection(client, collection_name):
    return client.collections.create(
        collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE  # select prefered distance metric
        ),
    )


