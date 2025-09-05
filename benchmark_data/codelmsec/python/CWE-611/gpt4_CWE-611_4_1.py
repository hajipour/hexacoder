from flask import Flask, request
import lxml.etree

app = Flask(__name__)

@app.post("/login")
def process_login():
    '''
    Implementation of an API endpoint taking an xml file from request.
    Extract userid and password process them, response based on condition.
    '''
