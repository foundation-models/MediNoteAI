import os
from medinote import initialize
from medinote.embedding.vector_search import execute_query



# Initialize base setting and config file
main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or f"{os.path.dirname(__file__)}/..",
)


def table_similarity_search(rid=None):
    config = main_config.get('table_similarity_search')
    config["logger"] = logger

    # 1. Create the similarity table to save the informations:
    if config.get('recreate'):
        _ = execute_query(config.get('create_table'))

    # 2. find similarity 
    if rid is not None:
        matches = execute_query(config.get('search'), (str(rid),))
    else:
        matches = execute_query(config.get('search'))

    for match in matches:
        _ = execute_query(config.get('save_result'), (match))
    
    

if __name__ == "__main__":
    table_similarity_search()