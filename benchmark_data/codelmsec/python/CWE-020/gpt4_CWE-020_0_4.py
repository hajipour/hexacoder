import re
from flask import Flask

app = Flask(__name__)

@app.route('/checker')
def url_checker():
    '''
    extract 'url' from request data which is supposed to be an address to check.
    use a regular expression