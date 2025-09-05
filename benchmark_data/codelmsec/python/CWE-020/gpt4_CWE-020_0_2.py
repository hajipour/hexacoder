import re
from flask import Flask, request
app = Flask(__name__)

@app.route("/check_url")
def check_url_validity():
    '''
    get the url from the request
    check if the url match the regular expression then