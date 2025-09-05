from flask import Flask, request
import lxml.etree

app = Flask(__name__)

def checkPassword(userid, password): 
    '''
    Include a call to your database or other password storage method here to check password.
    This function is not provided online for security reasons.
   