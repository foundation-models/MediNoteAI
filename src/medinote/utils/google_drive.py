import os
import pickle
import gdown
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request

# Define the scope for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class GoogleDriveDownloader:
    def __init__(self, token_path, target_folder):
        self.token_path = token_path
        self.target_folder = target_folder

        # Authenticate and create the Google Drive API client
        self.service = self.authenticate_google_drive()

    def authenticate_google_drive(self):
        """Authenticate and create the Google Drive API client."""
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first time.
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)  # Use the 'credentials.json' for initial authentication
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        service = build('drive', 'v3', credentials=creds)
        return service

    def list_files_in_folder(self, folder_id):
        """Lists all files in a given Google Drive folder."""
        try:
            query = f"'{folder_id}' in parents"
            results = self.service.files().list(q=query, fields="files(id, name)").execute()
            files = results.get('files', [])
            return files
        except HttpError as error:
            print(f"An error occurred: {error}")
            return []

    def get_all_documents(self, docs):
        """Download all files inside the provided Google Drive folder into the target folder, 
        only if they are not already in the target folder."""
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)

        for doc in docs:
            if 'https://drive.google.com' in doc:
                folder_id = doc.split('/')[-1]  # Extract folder ID from the URL

                # Step 1: List all files in the folder
                drive_files = self.list_files_in_folder(folder_id)
                if not drive_files:
                    print(f"No files found in the folder: {doc}")
                    continue

                # Step 2: Get existing files in the target folder
                existing_files = set(os.listdir(self.target_folder))

                # Step 3: Download missing files
                for drive_file in drive_files:
                    file_name = drive_file['name']
                    if file_name not in existing_files:
                        print(f"Downloading {file_name}...")
                        file_id = drive_file['id']
                        file_url = f"https://drive.google.com/uc?id={file_id}"
                        gdown.download(file_url, os.path.join(self.target_folder, file_name), quiet=False)
                    else:
                        print(f"File {file_name} already exists, skipping download.")

# Example Usage
if __name__ == "__main__":
    # Define paths and parameters
    target_folder = 'docs'
    token_path = 'token.pickle'  # Path to the saved token file
    
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
