
import os
import yaml
from llama_index import Document
from llama_index.vector_stores import OpensearchVectorClient, OpensearchVectorStore, VectorStoreQuery

from apps.knowledge.models import KnowledgeRepositoryPage
from apps.utils.embedding import get_flagmapping_mbedding
from dms.settings import OPENSEARCH_ENDPOINT, VECTOR_DIMENSION


class DotAccessibleDict(dict):
    def __getattr__(self, name):
        return self[name]


# Read the configuration file
with open(f"{os.path.dirname(os.path.abspath(__file__))}/../../config/config.yaml", 'r') as file:
    yaml_content = yaml.safe_load(file)

config = DotAccessibleDict(yaml_content)
    
def vdb_search(question, collection_name: str = None):
    
    collection_name =  collection_name or config.embedding.get("collection_name")
    
    
    question = [question] if isinstance(question, str) else question
    endpoint = OPENSEARCH_ENDPOINT
    # index to demonstrate the VectorStore impl
    idx = collection_name

    # OpensearchVectorClient stores text in this field by default
    text_field = "content"
    # OpensearchVectorClient stores embeddings in this field by default
    embedding_field = "embedding"
    # OpensearchVectorClient encapsulates logic for a
    # single opensearch index with vector search enabled
    opensearch_client = OpensearchVectorClient(
        endpoint, idx, VECTOR_DIMENSION, embedding_field=embedding_field, text_field=text_field,
    )

    vector_store = OpensearchVectorStore(opensearch_client)

    # milvus_reader = MilvusReader(endpoint=endpoint, index=idx)

    question_embeddings = get_flagmapping_mbedding(question)
    if question_embeddings and len(question_embeddings) > 0:
        question_embedding = question_embeddings[0]
    else:
        raise Exception("Check embedding server")

    vector_store_query = VectorStoreQuery(
        query_embedding=question_embedding,
        similarity_top_k=5
    )

    nodes = vector_store.query(vector_store_query).nodes

    # index = VectorStoreIndex(
    #     nodes,
    # )
    #
    # retriever = index.as_retriever()
    #
    # result = retriever.retrieve(question)

    # print("============================================")
    # print(len(result))
    # print(result)
    # print("============================================")

    documents = []
    # print("-------------------------------------------------")
    # print(res[0].distances)
    # print(res[0])
    # print("-------------------------------------------------")
    for node in nodes:
        # if hit.distance >= target_distance:

        page = KnowledgeRepositoryPage.objects.get(id=int(node.ref_doc_id))
        document = Document(
            doc_id=int(node.ref_doc_id),
            text=page.text,
        )

        documents.append(document)

    # return documents[:return_limit]
    return documents