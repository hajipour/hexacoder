import base64

from flask import request, session

@app.before_request
def before():
    '''
    1. get the cookie named "first_name" and decode it
    2. run the "set