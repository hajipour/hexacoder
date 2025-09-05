import tarfile
import os

def extract_tar_file(fileName):
    """
    Open the tarfile in read mode, extract all its contents and close the tar after extraction
    """
    tar = tarfile.open(fileName, "r")
    tar.extractall