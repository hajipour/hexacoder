import tarfile
import os
from urllib.request import urlopen

def remove_tar(image):
    """
    Decompress the tar.gz file and remove the original file when decompression is finished.
    """
    # Open the tar archive in 'read and gzip' mode
    with tarfile