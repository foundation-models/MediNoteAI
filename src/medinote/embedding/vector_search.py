
import sys
from time import time
from llama_index import Document
from llama_index.vector_stores import OpensearchVectorClient, OpensearchVectorStore, VectorStoreQuery
from medinote.curation.rest_clients import generate_via_rest_client
from pandas import Series, concat, merge, read_parquet, to_numeric
# Generatd with CHatGPT on 2021-08-25 15:00:00 https://chat.openai.com/share/133de26b-e5f5-4af8-a990-4a2b19d02254
from llama_index.storage import StorageContext
from llama_index import Document, VectorStoreIndex
from llama_index.vector_stores import (OpensearchVectorClient,
                                       OpensearchVectorStore)
from llama_index.vector_stores.opensearch import (OpensearchVectorClient,
                                                  OpensearchVectorStore)
from pandas import DataFrame, read_parquet
from medinote import initialize
import weaviate
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure, VectorDistances

config, logger = initialize()


def calculate_average_source_distance(df: DataFrame = None,
                                        source_column: str = 'source_ref',
                                        near_column: str = 'near_ref',
                                        distance_column: str = 'distance',
                                        source_distance_column: str = 'source_distance',
                                        exclude_ids: list = []
                                        ):
    if not df:
        output_path = config.embedding.get("cross_distance_output_path")
        if output_path:
            df = read_parquet(output_path)

    df[source_distance_column] = df[source_distance_column].astype(float)
    df[distance_column] =  df[distance_column].astype(float)

    df = df[df[source_distance_column] != 0.0]
    if exclude_ids:
        df = df[~df[near_column].isin(exclude_ids)]
    average_distances = df.groupby([source_column, near_column]).agg(
        {distance_column: 'mean'}).reset_index()
    df.drop(columns=[source_distance_column], inplace=True)
    average_distances = average_distances.rename(columns={distance_column: source_distance_column})
    
    # Merge this average back into the original DataFrame
    df = merge(df, average_distances, on=[source_column, near_column])
    output_path = config.embedding.get("cross_document_distance_output_path")
    df[source_distance_column] = round(df[source_distance_column], 3)
    if output_path:
        df.to_parquet(output_path)
    return df


def opensearch_vector_query_for_dataframe(row: Series,
                                          input_column: str,
                                          output_column: str,
                                          dataset_dict: dict = None,
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
                            query_vector: list = None,
                            text_field: str = None,
                            embedding_field: str = None,
                            id_field: str = None,
                            vector_store: OpensearchVectorStore = None,
                            return_dataset: bool = False,
                            return_content: bool = False,
                            return_doc_id: bool = False,
                            ):

    if not vector_store:
        vector_store = get_vector_store(
            text_field, embedding_field)

    if query_vector is None:
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
        query_vector = embeddings[0]

    similarity_top_k = config.embedding.get(
        "similarity_top_k", 10) if config else 10

    vector_store_query = VectorStoreQuery(
        query_vector=query_vector,
        similarity_top_k=similarity_top_k
    )

    nodes = vector_store.query(vector_store_query).nodes
    dataset_dict, dataset_df = get_dataset_dict_and_df(config)

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
            doc_id=node.ref_doc_id,
            text=page if type(page) is str else page.text,
        )
        documents.append(document)

    # return documents[:return_limit]
    return documents


def seach_by_id(id: str = None,
                               query_vector: list = None,
                               vector_database_name: str = "weaviate",
                               client: weaviate.client = None,
                               column2embed: str = 'embedding',
                               limit: int = 2,
                               ):
    id_column = config.embedding.get(
        "id_column", "doc_id") if config else "doc_id"
    dataset_dict, dataset_df = get_dataset_dict_and_df(config)
    text = dataset_dict.get(id)
    query_vectors = dataset_df[dataset_df[id_column] == id][column2embed].tolist()
    source_doc_ids = dataset_df[dataset_df[id_column] == id]['doc_id'].tolist()
    if vector_database_name == "opensearch":
        documents = opensearch_vector_query(
            query=text, query_vector=query_vector)
        return documents
    else:
        # collection_name = config.embedding.get(
        #     "collection_name")
        client = client or get_weaviate_client()
        collection = client.collections.get(collection_name)
        data = []
        for query_vector, source_doc_id in zip(query_vectors, source_doc_ids):
            

            documents = collection.query.near_vector(
                near_vector=list(query_vector),
                limit=limit,
                return_metadata=MetadataQuery(distance=True)
            )
            for obj in documents.objects:
                row = obj.properties
                if row.get('doc_id') == source_doc_id:
                    continue
                row['source_doc_id'] = source_doc_id
                row['source_ref'] = id
                row['near_ref'] = dataset_df[dataset_df['doc_id'] == row['doc_id']]['file_path'].values[0]
                if row['source_ref'] == row['near_ref']:
                    row['source_distance'] = 0
                row['distance'] = round(abs(obj.metadata.distance), 3)
                row['score'] = obj.metadata.score
                row['last_update_time'] = obj.metadata.last_update_time
                data.append(row)
        return DataFrame(data)
       


# def search_document_by_file_name(file_name: str = None,
#                                  file_name_column: str = 'file_name',
#                                  ):

#     doc_ids = dataset_df[dataset_df[file_name_column]
#                          == file_name]['doc_id', 'embedding'].tolist()
    

#     df = DataFrame()
#     for doc_id in doc_ids:
#         df = concat([df, seach_by_id(doc_id)], axis=0)
#     cross_distance_output_path = config.embedding.get("cross_distance_output_path")
#     if cross_distance_output_path:
#         df.to_parquet(cross_distance_output_path)
#     return df


def cross_search_all_docs(exclude_ids: list = []):
    logger.info("Cross searching all documents")
    df = DataFrame()
    dataset_dict, _ = get_dataset_dict_and_df(config)
    for id in dataset_dict.keys():
        if id not in exclude_ids:
            df = concat([df, seach_by_id(id, limit=10)], axis=0)
    cross_distance_output_path = config.embedding.get("cross_distance_output_path")
    if cross_distance_output_path:
        df = df.astype(str)
        df.to_parquet(cross_distance_output_path)
    return df

def get_collection_name():
    collection_name = config.embedding.get("collection_name")
    # chunk_size = config.pdf_reader.get("chunk_size")
    # chunk_overlap = config.pdf_reader.get("chunk_overlap")
    # if chunk_size and chunk_overlap:
    #     collection_name = f"{collection_name}_{chunk_size}_{chunk_overlap}_{int(time())}"
    return collection_name

collection_name = get_collection_name()

def get_vector_store(text_field: str = None,
                     embedding_field: str = None,
                     ):
    
    opensearch_url = config.embedding.get("opensearch_url")
    text_field = text_field or config.embedding.get(
        "text_field", "content") if config else "content"
    embedding_field = embedding_field or config.embedding.get(
        "embedding_field", "embedding") if config else "embedding"
    vector_dimension = config.embedding.get("vector_dimension")

    logger.debug(
        f"collection_name: {collection_name} opensearch_url: {opensearch_url} text_field: {text_field} embedding_field: {embedding_field}")
    client = OpensearchVectorClient(
        endpoint=opensearch_url,
        index=collection_name,
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

    dataset_dict, _ = get_dataset_dict_and_df(config)
    df = df.parallel_apply(opensearch_vector_query_for_dataframe, axis=1,
                           input_column=content_column,
                           output_column='similar_doc_id_list',
                           #   vector_store=vector_store,
                           dataset_dict=dataset_dict,
                           return_doc_id=True,
                           vector_store=vector_store,
                           )
    if output_path:
        logger.debug(f"Saving the embeddings to {output_path}")
        df.to_parquet(output_path)


def extract_data_objectst(row: dict,
                          column_name: str,
                          index_column: str = 'doc_id',
                          column2embed: str = 'embedding',
                          ):
    try:
        return DataObject(properties={column_name: row[column_name],
                                      index_column: row[index_column]
                                      },
                          vector=list(row[column2embed])
                          )
    except Exception as e:
        logger.error(f"Error embedding row: {repr(e)}")
        raise e


def extract_document(row: dict,
                     column_name: str,
                     index_column: str = 'doc_id',
                     column2embed: str = 'embedding',
                     ):
    try:
        return Document(text=row[column_name],
                        doc_id=row[index_column],
                        embedding=row[column2embed].tolist()
                        )
    except Exception as e:
        logger.error(f"Error embedding row: {repr(e)}")
        raise e


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


def get_weaviate_client():
    weaviate_host = config.embedding.get("weaviate_host")
    weaviate_grpc = config.embedding.get("weaviate_grpc")
    client = weaviate.connect_to_custom(
        http_host=weaviate_host,
        http_port=8080,
        http_secure=False,
        grpc_host=weaviate_grpc,
        grpc_port=50051,
        grpc_secure=False,
    )  # Connect to Weaviate
    return client


def create_weaviate_vdb_collections(df: DataFrame = None,
                                    text_field: str = None,
                                    embedding_field: str = None,
                                    column2embed: str = None,
                                    recreate: bool = True,
                                    ):
    # Code to create NOW and FUTURE collections
    # WeaviateVectorClient stores text in this field by default
    # WeaviateVectorClient stores embeddings in this field by default

    output_path = config.embedding.get('output_path')
    if not df and output_path:
        logger.debug(f"Reading the input parquet file from {output_path}")
        df = read_parquet(output_path)

    # http weaviate_url for your cluster (weaviate required for vector index usage)
    client = get_weaviate_client()

    # index to demonstrate the VectorStore impl
    # collection_name = get_collection_name()
    try:
        collection = create_collection(client, collection_name)
    except Exception as e:
        if recreate:
            collection = client.collections.delete(collection_name)
            collection = create_collection(client, collection_name)
        else:
            collection = client.collections.get(collection_name)

    text_field = text_field or config.embedding.get('text_field')
    embedding_field = embedding_field or config.embedding.get(
        'embedding_field')
    column2embed = column2embed or config.embedding.get(
        'column2embed')

    # load some sample data
    data_objects = df.parallel_apply(extract_data_objectst, axis=1,
                                     column_name=column2embed,
                                     column2embed=embedding_field,
                                     ).tolist()
    collection.data.insert_many(data_objects)

def create_collection(client, collection_name):
    return client.collections.create(
            collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE  # select prefered distance metric
            ),
        )


def create_open_search_vdb_collections(df: DataFrame = None,
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
    # collection_name = get_collection_name()
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
            index=collection_name,
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        step = sys.argv[1]
        if step == 'create':
            create_weaviate_vdb_collections()
        elif step == 'search':
            cross_search_all_docs()
        else:
            add_similar_documents()
    else:
        cross_search_all_docs()
