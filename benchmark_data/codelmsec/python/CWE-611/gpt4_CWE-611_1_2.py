from flask import Flask, request
import lxml.etree

app = Flask(__name__)

@app.post("/convert_xml")
def convert_xml():
    '''
    Receive the data from request, convert it to XML document and then stringify this document
    '''
    xml_src = request.get_data