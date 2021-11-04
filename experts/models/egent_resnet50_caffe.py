"""Script to downlaod ImageNet pretrained weights from Google Drive

Extra packages required to run the script:
    colorama, argparse_color_formatter
"""

import argparse
import os
import requests



# ---------------------------------------------------------------------------- #
# Mapping from filename to google drive file_id
# ---------------------------------------------------------------------------- #
PRETRAINED_WEIGHTS = {
    'resnet50_caffe.pth': '1wHSvusQ1CiEMc5Nx5R8adqoHQjIDWXl1',
}


# ---------------------------------------------------------------------------- #
# Helper fucntions for download file from google drive
# ---------------------------------------------------------------------------- #

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def main():
    filename="resnet50_caffe.pth" 
    file_id = PRETRAINED_WEIGHTS[filename]
    destination = filename
    download_file_from_google_drive(file_id, destination)
    print('Download {} to {}'.format(filename, destination))


if __name__ == "__main__":
    main()
