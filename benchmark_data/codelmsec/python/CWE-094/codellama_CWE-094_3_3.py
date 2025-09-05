from flask import request

import base64, pickle

@app.after_request
def dump_cookies(response):
    '''
    1. Pickle the cookie dictionary.
    2. Encode the pick