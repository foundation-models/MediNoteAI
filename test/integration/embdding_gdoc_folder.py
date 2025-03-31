import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import io
from langchain_openai import OpenAIEmbeddings
import tempfile
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_path

from medinote import initialize, chunk_process, read_dataframe
from medinote.inference.inference_prompt_generator import row_infer

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

config = main_config.get(__file__.split("/")[-1].split(".")[0])

def extract_text_with_ocr(file_path):
    try:
        pages = convert_from_path(file_path)
        for i, page in enumerate(pages):
            text = pytesseract.image_to_string(page)
            print(f"--- Page {i+1} ---")
            print(text if text else "No text found on this page.")
    except Exception as e:
        print(f"Error during OCR: {e}")

# Function to read structural elements from Google Docs
def read_structural_elements(elements):
    """Recurses through a list of Structural Elements to read a document's text."""
    text = ''
    for value in elements:
        if 'paragraph' in value:
            elements = value.get('paragraph').get('elements')
            for elem in elements:
                text += read_paragraph_element(elem)
        elif 'table' in value:
            # The text in table cells
            table = value.get('table')
            for row in table.get('tableRows'):
                cells = row.get('tableCells')
                for cell in cells:
                    text += read_structural_elements(cell.get('content'))
        elif 'tableOfContents' in value:
            # The text in the TOC
            toc = value.get('tableOfContents')
            text += read_structural_elements(toc.get('content'))
    return text

def read_paragraph_element(element):
    """Returns the text in the given ParagraphElement."""
    text_run = element.get('textRun')
    if not text_run:
        return ''
    return text_run.get('content')

        
# Set up Google authentication
SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/documents.readonly'
]
creds = service_account.Credentials.from_service_account_file(
    config.get('gdoc_credentials'), scopes=SCOPES)

drive_service = build('drive', 'v3', credentials=creds)
docs_service = build('docs', 'v1', credentials=creds)

# Specify the folder ID
folder_id = config.get('root_folder_id')

# List files in the folder
query = f"'{folder_id}' in parents and trashed = false"
results = drive_service.files().list(
    q=query,
    pageSize=1000,
    fields="nextPageToken, files(id, name, mimeType)"
).execute()

items = results.get('files', [])

documents = []

try:
    for item in items:
        file_id = item['id']
        file_name = item['name']
        mime_type = item['mimeType']
        metadata = {'source': file_name}

        if mime_type == 'application/vnd.google-apps.document':
            # Get content from Google Doc
            doc = docs_service.documents().get(documentId=file_id).execute()
            content = read_structural_elements(doc.get('body').get('content'))
            documents.append(Document(page_content=content, metadata=metadata))
        elif mime_type == 'application/pdf':
            # Download the PDF file
            request = drive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)

            # Load the PDF from bytes
            # Step 2: Save the BytesIO content to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
                temp_file.write(fh.read())
                temp_file.flush()  # Ensure all data is written
                
                try:

                    # Step 3: Load the PDF using PyPDFLoader
                    loader = PyPDFLoader(temp_file.name)
                    pdf_documents = loader.load()

                    # Check if the loaded documents have content
                    if not pdf_documents or all(not doc.page_content.strip() for doc in pdf_documents):
                        raise ValueError("Empty content detected, initiating OCR process.")

                    for doc in pdf_documents:
                        # Add metadata
                        doc.metadata.update(metadata)
                    documents.extend(pdf_documents)
                except Exception as e:
                    print(f"Standard PDF loading failed: {e}. Running OCR...")

                    # Step 4: Run OCR extraction on the PDF
                    # ocr_text = extract_text_with_ocr(temp_file.name)

                    # if ocr_text:
                    #     # Create a new Document with the OCR-extracted text and metadata
                    #     ocr_doc = Document(page_content=ocr_text, metadata=metadata)
                    #     documents.append(ocr_doc)
                    #     print("OCR extraction successful and document added.")
                    # else:
                    #     print("OCR extraction failed or returned empty content.")
        else:
            # Skip other file types
            continue
except TypeError as e:
    print(f"Ignoring Error loading PDF: {e} .... and move on")

# Set up the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Split the documents into chunks
docs_splits = text_splitter.split_documents(documents)

# Set up the embeddings
embeddings = OpenAIEmbeddings(
    model="models/stella_en_400M_v5",
    openai_api_base="http://stella-en-400m-v5:8000",
    openai_api_key="YOUR_API_KEY",
)

# Get the texts to embed
texts = [doc.page_content for doc in docs_splits]

# Calculate embeddings
embeddings_list = embeddings.embed_documents(texts)

df = chunk_process(
    df=df,
    function=row_infer,
    config=config,
    chunk_size=100,
)


# Collect data into DataFrame
data = []
for i, doc in enumerate(docs_splits):
    data.append({
        'text': doc.page_content,
        'embedding': embeddings_list[i],
        'file_name': doc.metadata.get('source', ''),
        'page_number': doc.metadata.get('page', ''),
        # Add any other metadata fields as needed
    })

df = pd.DataFrame(data)

# Display the DataFrame
print(df.head())
