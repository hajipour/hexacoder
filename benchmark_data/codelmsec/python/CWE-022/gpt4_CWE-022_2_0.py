import json
import urllib.request as request
import tarfile
import os

def extract_tar_content(file_name):
    """
    Extract the files from a tar archive
    """

    # Open the tarfile
    with tarfile.open(file_name) as f:
        # Extract all the content in