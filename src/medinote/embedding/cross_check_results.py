import os
from pandas import DataFrame
from medinote import initialize, read_dataframe, write_dataframe

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

def cross_check_results(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(cross_check_results.__name__)
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
    text_column = config.get('text_column', 'text')
    truth_feature_column = config.get('truth_feature_column', 'text')
    truth_label_column = config.get('truth_label_column', 'label')
    predicted_column = config.get('predicted_column', 'predicted')
    df_truth = read_dataframe(config.get("truth_path"))
    df_truth.rename(columns={truth_feature_column: text_column, truth_label_column: 'actual'}, inplace=True)
    df = df.merge(df_truth, on=text_column)
    matching_percentage = (df[predicted_column] == df['actual']).mean() * 100
    logger.info(f'Matching percentage: {matching_percentage:.2f}%')
    output_path = config.get("output_path")
    if output_path:
        write_dataframe(df, output_path)
    return df

if __name__ == "__main__":
    cross_check_results()