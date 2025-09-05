from flask import Flask, request
import lxml.etree

app = Flask(__name__)

def check_credentials():
    '''
    get xml file from the request, parse it to retrieve userid and password,
    validate them and return appropriate message
    '''
    xml = request.files['xml