import tarfile
import os
from urllib.request import urlopen

def extract_and_remove_tar(image):
    '''
    Extracts the content of a tar archive and removes the archive file
    '''
    # Open the tar gzip archive
    with tarfile.open(image, mode='r: