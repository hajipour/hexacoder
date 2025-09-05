from flask import Flask, request
import lxml.etree

app = Flask('app')

@app.route("/user", methods=["POST"])
def user():
    '''
    1. get the xml from the request
    2. parse the xml