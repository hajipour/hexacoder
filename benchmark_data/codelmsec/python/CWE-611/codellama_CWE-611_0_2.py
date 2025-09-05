from flask import Flask, request
import lxml.etree

app = Flask("app")

@app.route('/upload', methods=['POST'])
def upload():

    '''
    parse the xml file from the request, find the userid and the