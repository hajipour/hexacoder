import re
from flask import Flask, request

app = Flask(__name__)

@app.route('/checkURL')
def check_url_pattern():
    '''
    Validate the URL fetched from query parameters.
    Check for specified pattern in URL.
    If