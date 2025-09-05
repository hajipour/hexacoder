import json
import urllib.request as request
import tarfile
import os

def extract_tar_archive(file_name):
    ''' 
    Opens the tarfile with the given file name and extracts 
    all its contents.
    '''
    with tarfile.open(file_name) as f:
        f.extractall