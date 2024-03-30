from pandas import DataFrame, Series, concat, read_parquet
from sklearn.metrics import classification_report

from medinote import initialize
from medinote.cached import read_dataframe, write_dataframe
import os

config, logger = initialize()


def measure_metrics(
    df: DataFrame = None,
    true_label: str = None,
    pred_label: str = None,
    label_names_column: str = None,
    default_true_value: str = None,
    experiment_name: str = None,
    dropped_feature_ids: list = [],
    persist: bool = True,
):
    if df is None:
        input_path = config.classification["metrics_input_path"]
        df = read_dataframe(input_path)

    pred_label = pred_label or config.classification.get("pred_label")
    true_label = true_label or config.classification.get("true_label")
    label_names_column = (
        label_names_column or config.classification.get("label_names_column") or true_label
    )
    default_true_value = default_true_value or config.classification.get("default_true_value")

    y_true = (
        df[true_label]
        if true_label in df.columns
        else Series([default_true_value] * len(df))
    )

    y_pred = df[pred_label]
    # Check if y_pred contains any None values
    if y_pred.isnull().any():
        y_pred = y_pred.fillna(default_true_value)

    target_names = (
        df[label_names_column].unique()
        if label_names_column and label_names_column in df.columns
        else None
    )
    # report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report = classification_report(y_true, y_pred, output_dict=True)
    df = DataFrame(report).transpose()
    experiment_name = experiment_name or config.classification.get("experiment_name") or "default"
    experiment_name = f"{experiment_name}_{len(dropped_feature_ids)}"
    df["experiment_name"] = experiment_name


    output_path = config.classification.get("metrics_output_path")
    # df_out = None
    if persist and output_path:
        write_dataframe(df=df, 
                        output_path=output_path, 
                        do_concat=True)
    #     df_out = (
    #         read_dataframe(output_path) if os.path.exists(output_path) else DataFrame()
    #     )        
    # if persist and output_path:
    #     df.to_csv(output_path, index=False)
    return df


def detect_label(
    get_true_label_function: callable,
    get_feature_id_function: callable,
    df: DataFrame = None,
    dropped_feature_ids: list = [],
    feature: str = None,
    true_label: str = None,
    pred_label: str = None,
    feature_id: str = None,
    experiment_name: str = None,
    persist: bool = True,
):

    feature = feature or config.classification.get("feature")
    true_label = true_label or config.classification.get("true_label")
    pred_label = pred_label or config.classification.get("pred_label")
    feature_id = feature_id or config.classification.get("feature_id")
    experiment_name = experiment_name or config.classification.get("experiment_name") or "default"
    experiment_name = f"{experiment_name}_{len(dropped_feature_ids)}"

    if df is None:
        input_path = config.classification.get("input_path")
        if not input_path:
            raise ValueError(f"No input_path found.")
        df = read_parquet(input_path)
    if df.empty:
        logger.info(f"No rows found in the detect_label DataFrame.")
        return
    # df[true_label] = df[feature].apply(get_true_label_function)
    df[feature_id] = df[feature].apply(get_feature_id_function)

    unique_counts = df.groupby(feature_id)[pred_label].nunique()
    more_than_one = unique_counts[unique_counts > 1]
    assert (
        more_than_one.empty
    ), f"More than one candidate label found for the following feature: {more_than_one}"
    df_grouped = df.groupby([feature_id, true_label])[pred_label].first().reset_index()
    
    df_grouped["dropped_feature_size"] = len(dropped_feature_ids)
    df_grouped["dropped_feature_ids"] = dropped_feature_ids if dropped_feature_ids else None

    # df['match'] = df['source_folder'] == df[pred_label]

    # match_list = df.groupby(feature)['match'].mean()
    # logger.info(match_list)
    # df = match_list.reset_index()
    # df.columns = [name, 'matched']
    # df[name] = df[name].apply(lambda path: os.path.basename(path))
    # df['matched'] = df['matched'] == 1.0

    output_prefix = config.classification.get("output_prefix")
    if persist and output_prefix:
        logger.info(f"Saving to {output_prefix}_{experiment_name}.parquet")
        df_grouped.to_parquet(f"{output_prefix}_{experiment_name}.parquet")

    # true_percentage = (df['matched'] == True).mean() * 100
    # logger.info(f"Percentage of True values in df['matched']: {true_percentage:.2f}%")
    # return true_percentage
    return df_grouped


if __name__ == "__main__":
    measure_metrics()
