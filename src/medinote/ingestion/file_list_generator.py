import os
from pandas import DataFrame
from medinote import initialize, read_dataframe

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)


def is_pdf(file_path) -> bool:
    try:
        with open(file_path, 'rb') as file:
            header = file.read(4)
            return header == b'%PDF'
    except Exception as e:
        return False


def file_list_generator(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(file_list_generator.__name__)
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
    assert df is None
    root_folder = config.get("root_folder")
    assert root_folder is not None
    logger.info(f"Getting all files in {root_folder}")
    file_list = get_all_files(root_folder)
    df = DataFrame(file_list, columns=["file_path"])
    if output_path := config.get("output_path"):
        logger.info(f"Writing to {output_path}")
        df.to_csv(output_path, index=False)
    return df


def get_all_files(root_folder):
    file_list = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if is_pdf(file_path):  # accept pdfs only
                file_list.append(file_path)
    return file_list


if __name__ == "__main__":
    file_list_generator()
