#pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
import mimetypes
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from _classes.Utility import ReadConfigString 
import io, os
FOLDER_ID = "1smkW4lWnT2jyO3BLVSq0XYRgU9hQNiKk"

def GetCreds():
	cred_file = r"data\google_api\credentials.json"
	token_file = r"data\google_api\token.json"
	SCOPES = ["https://www.googleapis.com/auth/drive.file"]
	creds = None
	if os.path.exists(token_file):
		creds = Credentials.from_authorized_user_file(token_file, SCOPES)
	if not creds or not creds.valid:
		if creds and creds.expired and creds.refresh_token:
			creds.refresh(Request())   # auto refresh silently
		else:
			flow = InstalledAppFlow.from_client_secrets_file(cred_file, SCOPES)
			creds = flow.run_local_server(port=0)
		with open(token_file, "w") as token:
			token.write(creds.to_json())
	return creds

def GoogleAPIUploadFile(file_path):
	if not os.path.exists(file_path):
		print(f"File not found: {file_path}")
	else:
		creds = GetCreds()
		drive_service = build('drive', 'v3', credentials=creds)
		file_name = os.path.basename(file_path)
		query = f"name = '{file_name}' and '{FOLDER_ID}' in parents and trashed = false"
		response = drive_service.files().list(q=query, fields='files(id)').execute()
		files = response.get('files', [])
		mime_type, _ = mimetypes.guess_type(file_path)
		media = MediaFileUpload(file_path, mimetype=mime_type)
		if files:			
			file_id = files[0]['id']
			updated_file = drive_service.files().update(
				fileId=file_id,
				media_body=media
			).execute()
			print(f"Updated existing file ID: {file_id}")
		else:
			file_metadata = {'name': file_name, 'parents': [FOLDER_ID]}
			new_file = drive_service.files().create(
				body=file_metadata,
				media_body=media,
				fields='id'
			).execute()
			print(f"Created new file ID: {new_file.get('id')}")

def GoogleAPIWriteReport(contents_text:str, file_name:str = 'report.txt'):
	creds = GetCreds()
	drive_service = build('drive', 'v3', credentials=creds)
	file_metadata = {"name": file_name, "parents": [FOLDER_ID] }
	buffer = io.BytesIO(contents_text.encode("utf-8"))
	media = MediaIoBaseUpload(
		buffer,
		mimetype="text/plain"
	)
	file = drive_service.files().create(
		body=file_metadata,
		media_body=media,
		fields="id"
	).execute()
	print("Uploaded:", file["id"])

	