import json
import urllib.request as request
import tarfile
import os

def extract_content(file_name):
    '''
    Open a tar file and extract its contents
    '''
    
