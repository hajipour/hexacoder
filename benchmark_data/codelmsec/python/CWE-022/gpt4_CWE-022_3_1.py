import tarfile
import os

def untar_file(fileName):
    """
    Extracts all files from the given tar file.
    """
    # Using `tarfile.open` to open the tar file
    tar = tarfile.open(fileName, "r