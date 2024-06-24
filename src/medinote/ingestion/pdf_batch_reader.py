import os
from PyPDF2 import PdfReader
from llama_index.core.node_parser import SentenceSplitter
from pandas import DataFrame
import yaml
from medinote import initialize, read_dataframe, chunk_process

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or f"{os.path.dirname(__file__)}/../..",
)

def pdf_batch_reader(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(pdf_batch_reader.__name__)
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
    df.rename(columns={"text": "file_path"}, inplace=True)
    df = chunk_process(
        df=df,
        function=pdf_reader,
        config=config,
        chunk_size=20,
    )
    return df

def pdf_reader(row: dict, config: dict):
    if row is None or not row.get("file_path"):
        return DataFrame()
    try:
        reader = PdfReader(row["file_path"])
    except Exception as e:
        logger.error(f"Error reading {row['file_path']}: {e}")
        logger.error(f"Skipping {row['file_path']}")
    pages = config.get("pages", [1, len(reader.pages)])
    chunk_size = config.get("chunk_size", 100)
    chunk_overlap = config.get("chunk_overlap", 10)
    page_range = range(pages[0], pages[1] + 1)

    texts = []
    for page_num, page in enumerate(reader.pages, 1):
        if page_num in page_range:
            text = page.extract_text()
            texts.append(text)
    if texts:
        if chunk_size == -1:
            chunks = ["\n".join(texts)]
        else:
            splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = splitter.split_texts(texts)
    else:
        chunks = []
    df = DataFrame(chunks, columns=["text"])
    for key, value in row.items():
        df[key] = value
    return df

if __name__ == "__main__":
    pdf_batch_reader()