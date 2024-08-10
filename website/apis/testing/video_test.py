import requests
import os

API_URL = "http://127.0.0.1:5000/api/analyze"
UPLOAD_FILE_PATH = "img1.jpg"  # Replace with the actual path

def upload_file(file_path):
    with open(file_path, 'rb') as file:
        response = requests.post(API_URL, files={'file': file})
    return response.json()

def main():
    response = upload_file(UPLOAD_FILE_PATH)
    print("Response from API:", response['result'])

if __name__ == "__main__":
    main()
