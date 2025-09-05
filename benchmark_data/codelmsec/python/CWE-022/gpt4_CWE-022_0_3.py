import os
import tarfile

def extract_tar_files(tar_path, dest_dir):
    '''
    Open the provided tar archive file and extract each of the files it contains.
    These files are placed in a specified destination directory.
    After extraction, the