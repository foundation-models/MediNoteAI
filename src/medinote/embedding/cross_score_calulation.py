import os
from numpy import vstack
from pandas import DataFrame
from medinote import initialize, read_dataframe, chunk_process, write_dataframe

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or f"{os.path.dirname(__file__)}/..",
)

def cross_score_calulation(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(cross_score_calulation.__name__)
    config["logger"] = logger
    df = (
        df
        if df is not None
        else (
            read_dataframe(config.get("input_path"))
            if config.get("input_path")
            else None
        )
    )
    query_condition = config.get('query_condition', 'is_query == True')
    df_query = df.query(query_condition).reset_index(drop=True)
    df_passage = df.query(f'not ({query_condition})').reset_index(drop=True)
    matches = cross_scores(df_query=df_query, df_passage=df_passage)
    df_query['match'] = None
    for i, row in matches.iterrows():
        df_query.loc[row['df_query_index'], 'match'] = df_passage.loc[row['df_passage_index'], 'text']
        df_query.loc[row['df_query_index'], 'score'] = row['score']
    df = df_query
    output_path = config.get("output_path")
    if output_path:
        write_dataframe(df, output_path)
    return df

def cross_scores(df_query, df_passage):
    # Convert embeddings to NumPy arrays for efficient computation
    embeddings1 = vstack(df_query['embedding'].values)
    embeddings2 = vstack(df_passage['embedding'].values)

    # Calculate the scores
    scores = embeddings1 @ embeddings2.T * 100

    # Create a DataFrame for the scores
    scores_df = DataFrame(scores)

    # Find the highest score for each row in df_query
    highest_scores_indices = scores_df.idxmax(axis=1)
    highest_scores = scores_df.max(axis=1)

    # Create a match DataFrame
    matches = DataFrame({
        'df_query_index': range(len(df_query)),
        'df_passage_index': highest_scores_indices,
        'score': highest_scores
    })
    # Optional: Sort matches by score
    matches = matches.sort_values(by='score', ascending=False)

    # Reset index if needed
    matches.reset_index(drop=True, inplace=True)
    return matches

if __name__ == "__main__":
    cross_score_calulation()