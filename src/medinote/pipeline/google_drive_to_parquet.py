import gdown
import os
import pymupdf
import pandas as pd
import yaml
import json
import pyarrow.parquet as pq
import pyarrow as pa
import logging
import requests
import numpy as np
from medinote import initialize, read_dataframe, chunk_process
from medinote.inference.ollama_api import row_infer
from medinote.utils.google_drive import GoogleDriveDownloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from YAML file
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
config = config.get('embedding_generator')
config['logger'] = logging


def split_text_with_overlap(text, batch_size, overlap_percentage):
    """Splits text into chunks with overlap."""
    chunks = []
    step = int(batch_size * (1 - overlap_percentage))

    for i in range(0, len(text), step):
        chunk = text[i:i + batch_size]
        chunks.append(chunk)
        if i + batch_size >= len(text):
            break

    return chunks


def read_all_pdfs(target_folder, batch_size, overlap_percentage):
    """Reads all PDF files in the target folder and processes text in chunks with overlap."""
    if not os.path.exists(target_folder):
        print(f"Folder '{target_folder}' does not exist.")
        return

    pdf_files = [f for f in os.listdir(target_folder) if f.endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the folder.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(target_folder, pdf_file)
        print(f"\nProcessing PDF: {pdf_file}\n{'-'*40}")
        if pdf_file.replace('.pdf','.parquet') in os.listdir(target_folder):
            print(f"Parquet file already exists for {pdf_file}. Skipping...")
            continue

        chunk_data = []
        prev_page_text = ""  # Store last portion of the previous page for inter-page overlap

        try:
            with pymupdf.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text("text")
                    text = text.replace('\n', ' ')  # Remove newlines

                    if prev_page_text:
                        text = prev_page_text + text  # Add overlap from previous page

                    chunks = split_text_with_overlap(text, batch_size, overlap_percentage)

                    for chunk in chunks:
                        chunk_json = {
                            "text": chunk,
                            "page": page_num,
                            "pdf_name": pdf_file
                        }
                        chunk_data.append(chunk_json)

                    overlap_chars = int(batch_size * overlap_percentage)
                    prev_page_text = text[-overlap_chars:] if len(text) > overlap_chars else text

            # Save extracted chunks as Parquet
            save_to_parquet(chunk_data, pdf_file, target_folder)

        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")


def save_to_parquet(chunk_data, pdf_file, target_folder):
    """Saves chunk data into a Parquet file with 'text' and 'data' columns."""
    if not chunk_data:
        print(f"No data to save for {pdf_file}. Skipping...")
        return

    df = pd.DataFrame(chunk_data)
    df["data"] = df.apply(lambda row: json.dumps({"page": row["page"], "pdf_name": row["pdf_name"]}), axis=1)
    df = df[["text", "data"]]

    parquet_filename = os.path.join(target_folder, f"{os.path.splitext(pdf_file)[0]}.parquet")

    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_filename)

    print(f"Saved Parquet file: {parquet_filename}")


if __name__ == "__main__":
    # Define paths and parameters
    target_folder       = 'docs'
    token_path          = 'token.pickle'  # Path to the saved token file
    batch_size          = 500  # Number of characters per chunk
    overlap_percentage  = 0.2  # 20% overlap
    
    docs = [
        'https://drive.google.com/drive/folders/13IDuCnHVO9Ral_lbCSUnOLg0C94Ti680'
    ]

    # Initialize GoogleDriveDownloader class
    downloader = GoogleDriveDownloader(
        token_path=token_path,
        target_folder=target_folder
    )

    # Run the downloader to fetch files
    downloader.get_all_documents(docs)

    # Process PDFs and save as Parquet as chunked data
    read_all_pdfs(target_folder, batch_size, overlap_percentage)

    # Process the Parquet file for inference (example of usage)
    for file_ in os.listdir(target_folder):
        if file_.endswith('.parquet'):
            config['output_prefix'] = f'./{file_.replace(' ','')}/' 
            parquet_file_path       = os.path.join(target_folder, file_)
            chunk_size              = 20
            df                      = pd.read_parquet(parquet_file_path)
            df                      = chunk_process(df=df, function=row_infer, config=config, chunk_size=chunk_size)

