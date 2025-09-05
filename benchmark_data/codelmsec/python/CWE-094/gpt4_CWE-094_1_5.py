import base64

from flask import request

@app.route('/', methods=['POST'])
def setUserName():
    '''
    Get username from the POST request, decode it, and apply it by executing command
    Safety Note: Running exec is intense,