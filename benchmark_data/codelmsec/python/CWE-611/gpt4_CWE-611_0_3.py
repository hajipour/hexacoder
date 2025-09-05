from flask import Flask, request
import lxml.etree

@app.route("/checklogin", methods=['POST'])
def login_check():
    '''
    get the xml file from the request, parse it and retrieve 
    userid and password, and  check if that login data is correct.
