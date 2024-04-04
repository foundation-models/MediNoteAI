from medinote import initialize, setup_logging
from medinote.augmentation.merger import (
    merge_all_screened_files,
    merge_all_sqlcoder_files,
)
from medinote.augmentation.sqlcoder import parallel_generate_synthetic_data
from medinote.curation.screening import api_screening
from medinote.embedding.vector_search import create_weaviate_vdb_collections


logger, _ = initialize()


def pipeline(config: dict = None):
    # be sure sqlcoder pod is running
    given_schema = config.get("schemas").get("deal")
    parallel_generate_synthetic_data("deal", given_schema=given_schema)
    merge_all_sqlcoder_files()
    api_screening()
    merge_all_screened_files()
    # parallel_generate_embedding()
    # merge_all_embedding_files()
    create_weaviate_vdb_collections()

