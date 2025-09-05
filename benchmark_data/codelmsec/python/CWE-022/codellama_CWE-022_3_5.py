import tarfile
import os

import json

def extract_json(json):
    '''
    Extract the json data from the json archive
    '''

    # Write your code here
    if tarfile.is_tar