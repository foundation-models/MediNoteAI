import os
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter

from llama_index.node_parser import SentenceSplitter
from medinote import initialize

config, logger = initialize()

# Define a function to read and split a PDF file
def process_pdf(file_path):
    reader = PdfReader(file_path)
    splitter = SentenceSplitter(
        chunk_size=200,
        chunk_overlap=15,
    )
    pages = (0, 1000)
    page_range = range(pages[0], pages[1] + 1)

    texts = []
    for page_num, page in enumerate(reader.pages, 1):
        if page_num in page_range:
            text = page.extract_text()
            texts.append(text)
    chunks = splitter.split_texts(texts)
    return [(chunk, file_path) for chunk in chunks]

# Function to process each PDF file
def process_pdf_wrapper(row: pd.Series):
    file_path = os.path.join(row['Root'], row['File'])
    rows = []
    for chunk in process_pdf(file_path):
        row_copy = row.copy()
        row_copy['text'] =  chunk[0]
        row_copy['file_path'] = chunk[1]
        rows.append(row_copy)
    return pd.DataFrame(rows)

# Traverse the folder and process each PDF file
def process_folder(folder_path):
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append((root, file))
    # Convert list to DataFrame
    df = pd.DataFrame(pdf_files, columns=['Root', 'File'])
    
    output_path = config.pdf_reader.get("output_path")

    chunk_size = 10
    num_chunks = len(df) // chunk_size + 1

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df))
        chunk_df = df[start_index:end_index]

        output_file = f"{output_path}_{start_index}_{end_index}.parquet" if output_path else None
        if output_file is None or not os.path.exists(output_file):
            try:
                chunk_df = pd.concat(chunk_df.parallel_apply(process_pdf_wrapper, axis=1).tolist(), ignore_index=True)

            except ValueError as e:
                if "Number of processes must be at least 1" in str(e):
                    logger.error(
                        f"No idea for error: Number of processes must be at least \n ignoring .....")
            except Exception as e:
                logger.error(f"Error generating synthetic data: {repr(e)}")

            if output_file:
                try:
                    chunk_df.to_parquet(output_file)
                except Exception as e:
                    logger.error(
                        f"Error saving the embeddings to {output_file}: {repr(e)}")
        else:
            logger.info(
                f"Skipping chunk {start_index} to {end_index} as it already exists.")



def read_pdf():
    # Define the folder path here
    folder_path = config.pdf_reader.get('input_path')

    # Process the folder and create the DataFrame
    data = process_folder(folder_path)
    df = pd.DataFrame(data, columns=['text', 'file_path'])

    # Show the DataFrame
    print(df.head())

if __name__ == '__main__':
    read_pdf()