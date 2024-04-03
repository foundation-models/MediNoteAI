from pandas import DataFrame, Series, concat, read_parquet
from sklearn.metrics import classification_report

from medinote import setup_logging
from medinote.cached import read_dataframe, write_dataframe
import os

logger = setup_logging()

def measure_metrics(
    df: DataFrame = None,
    true_label: str = None,
    pred_label: str = None,
    label_names_column: str = None,
    default_true_value: str = None,
    experiment_name: str = None,
    dropped_feature_length: int = 0,
    persist: bool = True,
    config: dict = None,
):
    input_path = config.get("classification").get("metrics_input_path")
    if df is None:
        df = read_dataframe(input_path)
        if not df.empty and "dropped_feature_length" in df.columns:
            df = df[df["dropped_feature_length"] == dropped_feature_length]
    elif not df.empty and persist:
        write_dataframe(df=df, output_path=input_path, do_concat=True)

    pred_label = pred_label or config.get("classification").get("pred_label")
    true_label = true_label or config.get("classification").get("true_label")
    label_names_column = (
        label_names_column
        or config.get("classification").get("label_names_column")
        or true_label
    )
    default_true_value = default_true_value or config.get("classification").get(
        "default_true_value"
    )

    # Compute the length of the dataframe if needed
    length = len(df)
    
    y_true = (
        df[true_label]
        if true_label in df.columns
        else Series([default_true_value] * length)
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
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    # report = classification_report(y_true, y_pred, output_dict=True)
    df = DataFrame(report).transpose()
    experiment_name = (
        experiment_name or config.get("classification").get("experiment_name") or "default"
    )
    df["experiment_name"] = experiment_name
    df["dropped_feature_length"] = dropped_feature_length
    df = df.reset_index()

    output_path = config.get("classification").get("metrics_output_path")
    # df_out = None
    if persist and output_path:
        write_dataframe(df=df, output_path=output_path, do_concat=True)
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
    dropped_feature_ids: list = None,
    feature: str = None,
    true_label: str = None,
    pred_label: str = None,
    feature_id: str = None,
    experiment_name: str = None,
    persist: bool = True,
    config: dict = None,
):

    feature = feature or config.get("classification").get("feature")
    true_label = true_label or config.get("classification").get("true_label")
    pred_label = pred_label or config.get("classification").get("pred_label")
    feature_id = feature_id or config.get("classification").get("feature_id")
    experiment_name = (
        experiment_name or config.get("classification").get("experiment_name") or "default"
    )

    if df is None:
        input_path = config.get("classification").get("input_path")
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

    df_grouped["dropped_feature_length"] = len(dropped_feature_ids) if dropped_feature_ids is not None else 0
    df_grouped["dropped_feature_ids"] = (
        [dropped_feature_ids] * len(df_grouped) if dropped_feature_ids else None
    ) if dropped_feature_ids is not None else None

    # df['match'] = df['source_folder'] == df[pred_label]

    # match_list = df.groupby(feature)['match'].mean()
    # logger.info(match_list)
    # df = match_list.reset_index()
    # df.columns = [name, 'matched']
    # df[name] = df[name].apply(lambda path: os.path.basename(path))
    # df['matched'] = df['matched'] == 1.0

    output_prefix = config.get("classification").get("output_prefix")
    if persist and output_prefix:
        logger.info(f"Saving to {output_prefix}_{experiment_name}.parquet")
        write_dataframe(
            df=df_grouped,
            output_path=f"{output_prefix}_{experiment_name}.parquet",
            do_concat=True,
        )

    # true_percentage = (df['matched'] == True).mean() * 100
    # logger.info(f"Percentage of True values in df['matched']: {true_percentage:.2f}%")
    # return true_percentage
    return df_grouped


if __name__ == "__main__":
    measure_metrics()
