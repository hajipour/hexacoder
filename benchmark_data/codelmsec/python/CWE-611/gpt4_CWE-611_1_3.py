from flask import Flask, request
import lxml.etree

@app.route("/xml", methods=["POST"])
def convert_xml_to_string():
    '''
    Parse xml data from request and return xml document as a string
    '''
    xml_src = request.get_data()
    doc = lxml