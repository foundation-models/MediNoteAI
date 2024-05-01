from medinote import chunk_process, initialize
from medinote.augmentation.merger import (
    merge_all_screened_files,
    merge_all_sqlcoder_files,
)
from medinote.augmentation.sqlcoder import parallel_generate_synthetic_data
from medinote.curation.screening import api_screening, screeen_dataframes
from medinote.embedding.vector_search import create_or_update_weaviate_vdb_collection


_, logger = initialize()


def pipeline(config: dict = None):
    # be sure sqlcoder pod is running
    given_schema = config.get("schemas").get("deal")
    parallel_generate_synthetic_data("deal", given_schema=given_schema)
    merge_all_sqlcoder_files()
    df = chunk_process(
        df=df,
        function=screeen_dataframes,
        config=config,
    )   
    merge_all_screened_files()
        # df = chunk_process(
        #     df=df,
        #     function=row_infer,
        #     config=generate_embedding,
        # )    # merge_all_embedding_files()
    create_or_update_weaviate_vdb_collection()

