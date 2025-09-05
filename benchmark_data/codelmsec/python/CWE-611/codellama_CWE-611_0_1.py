from flask import Flask, request
import lxml.etree

app = Flask()

@app.route("/login", methods=['POST'])
def login():
    '''
        get the xml_src, parse it, check the userid and password and return