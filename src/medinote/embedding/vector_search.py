
from llama_index import Document
from llama_index.vector_stores import OpensearchVectorClient, OpensearchVectorStore, VectorStoreQuery
from medinote.curation.rest_clients import generate_via_rest_client
from pandas import Series, read_parquet
# Generatd with CHatGPT on 2021-08-25 15:00:00 https://chat.openai.com/share/133de26b-e5f5-4af8-a990-4a2b19d02254
from llama_index.storage import StorageContext
from llama_index import Document, VectorStoreIndex
from llama_index.vector_stores import (OpensearchVectorClient,
                                       OpensearchVectorStore)
from llama_index.vector_stores.opensearch import (OpensearchVectorClient,
                                                  OpensearchVectorStore)
from pandas import DataFrame, read_parquet
from medinote import initialize

config, logger = initialize()

def opensearch_vector_query_for_dataframe(row: Series,
                                          input_column: str,
                                          output_column: str,
                                          dataset_dict: dict = None,
                                          dataset_df: DataFrame = None,
                                          text_field: str = None,
                                          embedding_field: str = None,
                                          vector_store: OpensearchVectorStore = None,
                                          return_dataset: bool = False,
                                          return_content: bool = False,
                                          return_doc_id: bool = False,
                                          ):
    row[output_column] = opensearch_vector_query(
        row[input_column],
        dataset_dict=dataset_dict,
        dataset_df=dataset_df,
        text_field=text_field,
        embedding_field=embedding_field,
        vector_store=vector_store,
        return_dataset=return_dataset,
        return_content=return_content,
        return_doc_id=return_doc_id,
    )
    return row

def get_dataset_dict_and_df(config):
    dataset_parquet_path = config.embedding.get(
        "dataset_parquet_path") if config else None
    if not dataset_parquet_path:
        raise ValueError(
            "You must provide either a dataset_dict or a dataset_parquet_file")
    dataset_df = read_parquet(dataset_parquet_path)
    id_column = config.embedding.get(
        "id_column", "doc_id") if config else "doc_id"
    content_column = config.embedding.get(
        "column2embed", "content") if config else "content"
    dataset_dict = dataset_df.set_index(
        id_column)[content_column].to_dict()

    return dataset_dict, dataset_df


def opensearch_vector_query(query: str,
                            text_field: str = None,
                            embedding_field: str = None,
                            id_field: str = None,
                            vector_store: OpensearchVectorStore = None,
                            return_dataset: bool = False,
                            return_content: bool = False,
                            return_doc_id: bool = False,
                            dataset_dict: dict = None,
                            dataset_df: DataFrame = None,
                            ):
       
    if not vector_store:
        vector_store = get_vector_store(
            text_field, embedding_field)

    query = [query] if isinstance(query, str) else query
    logger.debug(f"query: {query}")

    # OpensearchVectorClient encapsulates logic for a
    # single opensearch index with vector search enabled

    payload = {"input": [query]}
    embedding_url = config.embedding.get('embedding_url')
    id_field = id_field or config.embedding.get(
        "id_column", "doc_id") if config else "doc_id"
    embeddings = generate_via_rest_client(payload=payload,
                                          inference_url=embedding_url
                                          )
    if isinstance(embeddings, str) and '{"error"' in embeddings:
        raise Exception("Embedding service is not available")
    query_embedding = embeddings[0]

    similarity_top_k = config.embedding.get(
        "similarity_top_k", 10) if config else 10

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=similarity_top_k
    )

    nodes = vector_store.query(vector_store_query).nodes

    if return_dataset and dataset_df:
        return dataset_df[dataset_df[id_field].isin([node.ref_doc_id for node in nodes])]
    elif return_content and dataset_dict:
        return [dataset_dict.get(node.ref_doc_id) for node in nodes]
    elif return_doc_id:
        return [node.ref_doc_id for node in nodes]

    documents = []
    for node in nodes:
        # if hit.distance >= target_distance:

        page = dataset_dict.get(node.ref_doc_id)
        document = Document(
            doc_id=int(node.ref_doc_id),
            text=page.text,
        )
        documents.append(document)

    # return documents[:return_limit]
    return documents



def get_vector_store(text_field: str = None, 
                     embedding_field:str = None,
                     ):
    opensearch_index = config.embedding.get("opensearch_index")
    opensearch_url = config.embedding.get("opensearch_url")
    text_field = text_field or config.embedding.get(
        "text_field", "content") if config else "content"
    embedding_field = embedding_field or config.embedding.get(
        "embedding_field", "embedding") if config else "embedding"
    vector_dimension = config.embedding.get("vector_dimension")

    logger.debug(
        f"opensearch_index: {opensearch_index} opensearch_url: {opensearch_url} text_field: {text_field} embedding_field: {embedding_field}")
    client = OpensearchVectorClient(
        endpoint=opensearch_url,
        index=opensearch_index,
        dim=vector_dimension,
        embedding_field=embedding_field,
        text_field=text_field
    )
    vector_store = OpensearchVectorStore(client)
    return vector_store


def add_similar_documents(df: DataFrame = None,
                          text_field: str = None,
                          embedding_field: str = None,
                          ):
    vector_store = get_vector_store(
        text_field, embedding_field)
    input_path = config.embedding.get('input_path')
    output_path = config.embedding.get('output_path')
    content_column = config.embedding.get(
        "column2embed", "content") if config else "content"
    if not df and input_path:
        logger.debug(f"Reading the input parquet file from {input_path}")
        df = read_parquet(input_path)

    dataset_dict, dataset_df = get_dataset_dict_and_df(config)
    df = df.parallel_apply(opensearch_vector_query_for_dataframe, axis=1,
                           input_column=content_column,
                           output_column='similar_doc_id_list',
                           #   vector_store=vector_store,
                           dataset_dict=dataset_dict,
                           dataset_df=dataset_df,
                           return_doc_id=True,
                           vector_store=vector_store,
                           )
    if output_path:
        logger.debug(f"Saving the embeddings to {output_path}")
        df.to_parquet(output_path)


def extract_document(row: dict,
                     column_name: str,
                     index_column: str = 'doc_id',
                     column2embed: str = 'embedding',
                     ):
    return Document(text=row[column_name], doc_id=row[index_column], embedding=row[column2embed].tolist())


def chunk_documents(documents, chunk_size):
    """ Divide documents into chunks based on length. """
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


def create_vector_db_collections(df: DataFrame = None,
                                 text_field: str = None,
                                 embedding_field: str = None,
                                 column2embed: str = None,
                                 ):
    # Code to create NOW and FUTURE collections
    # OpensearchVectorClient stores text in this field by default
    # OpensearchVectorClient stores embeddings in this field by default

    output_path = config.embedding.get('output_path')
    if not df and output_path:
        logger.debug(f"Reading the input parquet file from {output_path}")
        df = read_parquet(output_path)

   # http opensearch_url for your cluster (opensearch required for vector index usage)
    opensearch_url = config.embedding.get("opensearch_url")
    # index to demonstrate the VectorStore impl
    opensearch_index = config.embedding.get("opensearch_index")
    vector_dimension = config.embedding.get("vector_dimnesion")
    text_field = text_field or config.embedding.get('text_field')
    embedding_field = embedding_field or config.embedding.get(
        'embedding_field')
    column2embed = column2embed or config.embedding.get(
        'column2embed')
    chunk_size = config.embedding.get('chunk_size')

    # load some sample data
    documents = df.parallel_apply(extract_document, axis=1,
                                  column_name=column2embed,
                                  column2embed=embedding_field,
                                  ).tolist()

    if len(documents) > 0:
        # OpensearchVectorClient encapsulates logic for a
        # single opensearch index with vector search enabled
        client = OpensearchVectorClient(
            endpoint=opensearch_url,
            index=opensearch_index,
            dim=vector_dimension,
            embedding_field=embedding_field,
            text_field=text_field
        )
        # initialize vector store
        vector_store = OpensearchVectorStore(client)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        # initialize an index using our sample data and the client we just created
        document_chunks = chunk_documents(
            documents, chunk_size) if chunk_size else [documents]

        indexes = []
        for chunk in document_chunks:
            index = VectorStoreIndex.from_documents(
                documents=chunk, storage_context=storage_context
            )
            indexes.append(index)
        return indexes[-1]
        # index.set_index_id(f"{knowledge.id}")
        # index.storage_context.persist(persist_dir=f"{VDB_ROOT}/{knowledge.id}", vector_store_fname=vector_store_fname)
