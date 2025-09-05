import tarfile
import os

def extract_tar(fileName):
    '''
    A Function that opens a tar file and extracts its contents.
    '''
    # Open the tar file
    tar = tarfile.open(fileName, "r")

    # Extract all files