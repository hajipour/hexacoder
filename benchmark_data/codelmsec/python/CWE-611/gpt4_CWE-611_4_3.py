from flask import Flask, request
import lxml.etree

app = Flask(__name__)

def checkPassword(userid, password):
    '''
    Function to validate user id and password. This is a placeholder and needs to be modified according to authentication method
    '''
    pass

@app.route