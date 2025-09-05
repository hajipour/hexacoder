import re
from flask import Flask, request

app = Flask(__name__)

@app.route("/check")
def url_verification():
    '''
    Get url from request
    Using a regular expression, check if the url is valid and matches a