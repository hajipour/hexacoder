from flask import Flask, request
import lxml.etree

app = Flask(__name__)

def isAuthenticated():
    '''
    Extract userid and password from uploaded XML file. If the userid and password checks are successful, return OK. Otherwise, return ERROR.
    '''
    xml = request.files