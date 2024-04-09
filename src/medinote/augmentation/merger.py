from medinote import initialize, merge_parquet_files, remove_files_with_pattern
import sys
import os
import glob

from medinote import write_dataframe

_, logger = initialize()



def merge_all_sqlcoder_files(
    pattern: str = None,
    output_path: str = None,
    obj_name: str = None,
    config: dict = None,
):
    pattern = pattern or config.get("sqlcoder").get("output_prefix") + "*"

    output_path = output_path or config.get("sqlcoder").get("merge_output_path")

    if obj_name:
        pattern = pattern.replace("SRC", obj_name)
        output_path = output_path.replace("SRC", obj_name)

    logger.info(f"Merging all SQLCoder files to {output_path}")
    df = merge_parquet_files(pattern, identifier_index=1, identifier=obj_name)
    to_remove_columns = [
        "totalRecords",
        "__index_level_0__",
        "text:1",
        "__index_level_0__:1",
    ]
    for col in to_remove_columns:
        if col in df.columns:
            df = df.drop(col, axis=1)

    input_column = config.get("sqlcoder").get("input_column")
    df = df[~df[input_column].str.contains("do not know", case=False, na=False)]

    write_dataframe(df=df, output_path=output_path)
    remove_files_with_pattern(pattern)


def merge_all_screened_files(
    pattern: str = None,
    output_path: str = None,
    obj_name: str = None,
    config: dict = None,
):
    pattern = pattern or config.get("screening").get("output_prefix") + "*"
    output_path = output_path or config.get("screening").get("merge_output_path")
    df = merge_parquet_files(pattern, identifier=obj_name)
    logger.info(f"Merging all Screening files to {output_path}")
    write_dataframe(df=df, output_path=output_path)
    remove_files_with_pattern(pattern)


def merge_all_pdf_reader_files(
    pattern: str = None,
    output_path: str = None,
    config: dict = None,
):
    """
    Merges all PDF reader files matching the given pattern into a single Parquet file.

    Args:
        pattern (str, optional): The pattern used to match the PDF reader files. If not provided, it will use the default pattern defined in the configuration file.
        output_path (str, optional): The path where the merged Parquet file will be saved. If not provided, it will use the default output path defined in the configuration file.

    Returns:
        None
    """
    pattern = (
        pattern
        or config.get("pdf_reader").get("merge_pattern")
        or config.get("pdf_reader").get("output_prefix") + "*"
    )
    output_path = output_path or config.get("pdf_reader").get("merge_output_path")
    df = merge_parquet_files(pattern)
    logger.info(f"Merging all Screening files to {output_path}")
    write_dataframe(df=df, output_path=output_path)
    remove_files_with_pattern(pattern)


def merge_all_embedding_files(
    pattern: str = None, output_path: str = None, config: dict = None
):
    """
    Merges all embedding files matching the given pattern into a single DataFrame
    and saves it as a Parquet file.

    Args:
        pattern (str, optional): The pattern to match the embedding files. If not provided,
            it uses the default pattern specified in the configuration.
        output_path (str, optional): The path to save the merged DataFrame as a Parquet file.
            If not provided, it uses the default output path specified in the configuration.
    """
    pattern = pattern or config.get("embedding").get("output_prefix") + "*"
    output_path = output_path or config.get("embedding").get("output_path")
    df = merge_parquet_files(pattern)
    write_dataframe(df=df, output_path=output_path)
    remove_files_with_pattern(pattern)


def custom():
    pattern = "/mnt/datasets/archive/parquet/sqlcoder/sqlcoder_assetfi*.parquet"
    df = merge_parquet_files(pattern)
    df.to_parquet("/mnt/aidrive/datasets/sql_gen/tmp.parquet")
    remove_files_with_pattern(pattern)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        step = sys.argv[1]
        if step == "screening":
            merge_all_screened_files(obj_name="asset")
        elif step == "embedding":
            merge_all_embedding_files()
        elif step == "sqlcoder":
            merge_all_sqlcoder_files(obj_name="asset")
        elif step == "pdf_reader":
            merge_all_pdf_reader_files()
        else:
            print("Invalid step argument. Please choose 'screening' or 'sqlcoder'.")
    else:
        merge_all_screened_files()
        print("Please provide a step argument. Choose 'screening' or 'sqlcoder'.")
