import os
from pandas import DataFrame, read_parquet
from medinote import initialize, dynamic_load_function_from_env_varaibale_or_config
from medinote.augmentation.merger import merge_all_embedding_files, merge_all_pdf_reader_files, merge_all_screened_files, merge_all_sqlcoder_files
from medinote.augmentation.sqlcoder import parallel_generate_synthetic_data
from medinote.curation.pdf_reader import process_folder
from medinote.curation.screening import api_screening
from medinote.embedding.embedding_generation import parallel_generate_embedding
from medinote.embedding.vector_search import calculate_average_source_distance, create_weaviate_vdb_collections, cross_search_all_docs


config, logger = initialize()


def pipeline():
    # be sure sqlcoder pod is running
    given_schema = config.schemas.get("deal")
    parallel_generate_synthetic_data('deal', given_schema=given_schema)
    merge_all_sqlcoder_files()
    api_screening()
    merge_all_screened_files()
    # parallel_generate_embedding()
    # merge_all_embedding_files()
    create_weaviate_vdb_collections()
        
if __name__ == "__main__":
    pipeline()