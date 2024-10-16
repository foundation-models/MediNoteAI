import os
from medinote import initialize
from medinote.embedding.vector_search import execute_query


# Initialize base setting and config file
main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or f"{os.path.dirname(__file__)}/..",
)

config = main_config.get('table_similarity_search')

def table_similarity_search():
    # 1. Create the similarity table to save the informations:
    if config.get('recreate'):
        _ = execute_query(config.get('create_table'))

    # 2. find similarity
    matches = execute_query(config.get('search'))

    # 3.Insert data into note_icd_similarity table
    insert_query = """
    INSERT INTO note_icd_similarity (note_id, pid, rid, note_text, icd_code, icd_id, distance)
    VALUES (%s, %s, %s, %s, %s,%s, %s);
    """
    
    for match in matches:
        print(match)
        _ = execute_query(config.get('save_result'), (match))
        # _ = execute_query(insert_query, (match))


if __name__ == "__main__":
    table_similarity_search()