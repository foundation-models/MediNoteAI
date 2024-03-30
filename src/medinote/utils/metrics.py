from pandas import DataFrame, Series, concat
from sklearn.metrics import classification_report

from medinote import initialize
from medinote.cached import read_dataframe
import os

config, logger = initialize()


def measure_metrics(
    df: DataFrame = None,
    y_true_column: str = None,
    y_pred_column: str = None,
    label_names_column: str = None,
    default_true_value: str = None,
):
    if df is None:
        input_path = config.metrics["input_path"]
        df = read_dataframe(input_path)

    output_path = config.metrics.get("output_path")
    df_out = None
    if output_path:
        df_out = (
            read_dataframe(output_path) if os.path.exists(output_path) else DataFrame()
        )

    y_pred_column = y_pred_column or config.metrics.get("y_pred_column")
    y_true_column = y_true_column or config.metrics.get("y_true_column")
    label_names_column = (
        label_names_column or config.metrics.get("label_names_column") or y_true_column
    )
    default_true_value = default_true_value or config.metrics.get("default_true_value")

    y_true = (
        df[y_true_column]
        if y_true_column in df.columns
        else Series([default_true_value] * len(df))
    )

    y_pred = df[y_pred_column]
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
    df["source_file_name"] = os.path.basename(input_path)

    if df_out is not None:
        df_out = concat([df_out, df], ignore_index=True)
        df_out.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    measure_metrics()
