import json
import urllib.request as request
import tarfile
import os

def untar(file_name):
    """
    Extract all files from a tar compressed archive
    """
    with tarfile.open(file_name) as f:
        # Use the extractall method to extract all the files
       