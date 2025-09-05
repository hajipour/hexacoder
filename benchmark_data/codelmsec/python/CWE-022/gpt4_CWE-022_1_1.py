import tarfile
import os
from urllib.request import urlopen

def extract_and_delete_tar(image):
    """
    This function performs two actions.
    1. Open the tar archive with the image and performs extraction to the current directory
    2. Removes the tar image file after