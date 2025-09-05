import tarfile
import os
from urllib.request import urlopen

def download_and_extract_tar_gz(image):
    """
    This function opens an image file in read and gzip mode, 
    then extract all archived files and delete the tar.gz file.
    pavrameters:
    -