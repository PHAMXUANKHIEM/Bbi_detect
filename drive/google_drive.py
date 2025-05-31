import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from config import SCOPES

def get_drive_service():
    creds = None
    if os.path.exists('drive/token.json'):
        creds = Credentials.from_authorized_user_file('drive/token.json', SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file('drive/credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('drive/token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def upload_to_drive(file_path, folder_id=None):
    service = get_drive_service()
    file_metadata = {'name': os.path.basename(file_path)}
    if folder_id:
        file_metadata['parents'] = [folder_id]
    media = MediaFileUpload(file_path, mimetype='image/jpeg')
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webViewLink'
    ).execute()

    link = f"https://drive.usercontent.google.com/download?id={file.get('id')}&export=view&authuser=0"
    return link
