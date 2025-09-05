import tarfile
import os
from urllib.request import urlopen

def extract_and_remove(image):
    '''
    Open the tar.gz file, extract its content to the current directory 
    and then remove the tar.gz file
    '''
    # Open zip file
    with tarfile.open