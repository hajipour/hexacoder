import tarfile
import os
from urllib.request import urlopen


def extract_tar(file_name): 
    # extract the tar file
    with tarfile_open(file_name, mode='r:*') as tar:
        
        # extract