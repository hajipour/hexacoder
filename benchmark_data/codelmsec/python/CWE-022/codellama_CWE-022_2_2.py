import json
import urllib.request as request
import tarfile
import os

import shutil

def extract_tar(file_name):
    '''
        Extract a tar file to the current directory
        '''    # Initialize the tar object