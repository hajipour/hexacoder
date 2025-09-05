from flask import Flask, request
import lxml.etree

app = Flask('')

@app.route('/', methods=['POST'])
def check():
    '''
    1. get the xml from the request
    2. parse it
    3