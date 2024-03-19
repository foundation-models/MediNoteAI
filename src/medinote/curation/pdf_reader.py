import os
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter

from llama_index.node_parser import SentenceSplitter
from medinote import initialize
import spacy

config, logger = initialize()

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities_and_metadata(document):
    # Process the document
    doc = nlp(document)
    
    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Extract metadata (assuming metadata includes entities like DATE and ORG)
    metadata = {
        'dates': [ent.text for ent in doc.ents if ent.label_ == 'DATE'],
        'organizations': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
        # You can add more specific metadata extraction rules here
    }

    return entities, metadata



# Define a function to read and split a PDF file
def process_pdf(file_path: str,
                pages: tuple = (0, 1000),
                chunk_size: int = 300,
                chunk_overlap: int = 50):
    reader = PdfReader(file_path)
    page_range = range(pages[0], pages[1] + 1)

    texts = []
    for page_num, page in enumerate(reader.pages, 1):
        if page_num in page_range:
            text = page.extract_text()
            texts.append(text)
    if chunk_size == -1:
        chunks = ['\n'.join(texts)]
    else:
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_texts(texts)
    return [(chunk, file_path) for chunk in chunks]

import re

def extract_matter(text):
    # Define a regular expression pattern to find the matter
    # This pattern looks for the text following "following matter or proceeding:"
    pattern = r'following matter or proceeding: (.*?)  '
    
    # Search for the pattern in the text
    match = re.search(pattern, text)

    # If a match is found, return the matter
    if match:
        return match.group(1)
    else:
        return ""
    
# Function to process each PDF file
def process_pdf_wrapper(row: pd.Series,
                        pages: tuple = (0, 1000),
                        chunk_size: int = 300,
                        chunk_overlap: int = 50):
    file_path = os.path.join(row['Root'], row['File'])
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    if "template" in file_path.lower():
        topic = file_name.lower().split("template")[-1]
        row["text"] = f"This is a Template document called {file_name}"
        row["file_path"] = file_path
        return pd.DataFrame([row])  
    elif "retainer" in file_path.lower() or  "engagemen2t letter" in file_path.lower() or "engagemen2tletter" in file_path.lower():         
        text, file_path = process_pdf(file_path, chunk_size=-1)[0]
        if "retainer" in file_path.lower():
            _, metadata = extract_entities_and_metadata(text)
            replaced_text = f"Document related to "
            processed_orgs = set()
            for org in metadata['organizations']:
                org = org.replace("Inc.", "Inc").replace("Corporation", "Corp")
                if org.lower() not in processed_orgs and 'intapp' not in org.lower():
                    replaced_text += org 
                    replaced_text += " and "
                    processed_orgs.add(org.lower())
            matter = extract_matter(text)
            if matter:
                replaced_text += " Concerning the matter " + matter
        else:
            header = text.split('   ')[0] 
            replaced_text = f"An engagement letter called {file_name} with the following header: {header}"
            # _, metadata = extract_entities_and_metadata(short_text)
        # replaced_text = f"File {file_name} related to "
        logger.info(replaced_text)
        row["text"] = replaced_text
        row["file_path"] = file_path
        return pd.DataFrame([row])
    else:       
        rows = []
        for chunk in process_pdf(file_path, 
                                pages=pages, 
                                chunk_size=chunk_size, 
                                chunk_overlap=chunk_overlap
                                ):
            row_copy = row.copy()
            row_copy['text'] =  chunk[0]
            row_copy['file_path'] = chunk[1]
            rows.append(row_copy)
        return pd.DataFrame(rows)

# Traverse the folder and process each PDF file
def process_folder():
    pdf_files = []
    folder_path = config.pdf_reader.get('input_path')
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append((root, file))
    # Convert list to DataFrame
    df = pd.DataFrame(pdf_files, columns=['Root', 'File'])
    
    output_prefix = config.pdf_reader.get("output_prefix")
    pdf_chunk_size = config.pdf_reader.get("chunk_size", 300)
    chunk_overlap = config.pdf_reader.get("chunk_overlap", 50)

    chunk_size = 10
    num_chunks = len(df) // chunk_size + 1

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df))
        chunk_df = df[start_index:end_index]

        output_file = f"{output_prefix}_{start_index}_{end_index}.parquet" if output_prefix else None
        if output_file is None or not os.path.exists(output_file):
            try:
                chunk_df = pd.concat(chunk_df.parallel_apply(process_pdf_wrapper, axis=1,
                                                             chunk_size=pdf_chunk_size,
                                                             chunk_overlap=chunk_overlap,
                                                             ).tolist(), ignore_index=True)

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



if __name__ == '__main__':
    process_folder()