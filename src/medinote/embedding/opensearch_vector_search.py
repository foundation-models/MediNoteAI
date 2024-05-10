import os

from llama_index.vector_stores.types import VectorStoreQuery
from medinote import initialize
from medinote import write_dataframe
from medinote.curation.rest_clients import generate_via_rest_client
from pandas import DataFrame, Series, read_parquet

from medinote.embedding.vector_search import chunk_documents, extract_document, get_collection_name, get_dataset_dict_and_df

try:
    from llama_index import Document, VectorStoreIndex
except ImportError:
    from llama_index.core import Document, VectorStoreIndex

from llama_index.storage import StorageContext
from llama_index.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
)


_, logger = initialize()


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
        id_field = id_field or config.get("id_column", "doc_id") if config else "doc_id"
        embeddings = generate_via_rest_client(
            payload=payload, inference_url=embedding_url
        )
        if isinstance(embeddings, str) and '{"error"' in embeddings:
            raise Exception("Embedding service is not available")
        query_vector = embeddings[0]

    similarity_top_k = config.get("similarity_top_k", 10) if config else 10

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


def get_vector_store(
    text_field: str = None,
    column2embed: str = None,
    config: dict = None,
):

    opensearch_url = config.get("opensearch_url")
    text_field = (
        text_field or config.get("text_field", "content") if config else "content"
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
    content_column = config.get("column2embed", "content") if config else "content"
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
