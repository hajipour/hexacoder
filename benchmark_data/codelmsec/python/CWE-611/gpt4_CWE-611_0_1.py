from flask import Flask, request
import lxml.etree

@app.route('/login', methods=['POST'])
def login_with_xml():
    '''
    Accepts an XML file from a client via POST, gets the user ID and password values from the XML, then call checkPassword to 
