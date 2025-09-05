import json
import urllib.request as request
import tarfile
import os

def unpack_tarball(file_name):
    '''
    Given the tarball file name, the function opens and subsequently 
    extracts all files from it in the current directory.
    '''
    with tarfile.open(file_name) as