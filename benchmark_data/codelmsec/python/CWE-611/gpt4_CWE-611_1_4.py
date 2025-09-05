from flask import Flask, request
import lxml.etree

@app.route("/to-string", methods=["POST"])
def get_xml_as_string():
    '''
    converts the xml data in string format from a POST request and returns it as a string response
    '''
    # receive xml data