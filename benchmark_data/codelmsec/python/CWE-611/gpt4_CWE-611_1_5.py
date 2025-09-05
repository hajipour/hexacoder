from flask import Flask, request
import lxml.etree

@app.route("/parse", methods=["POST"])
def parse_and_return():
    '''
    receive XML data from POST request, parse and put it into string form
    '''
    xml_src = request.get_data()
    doc =