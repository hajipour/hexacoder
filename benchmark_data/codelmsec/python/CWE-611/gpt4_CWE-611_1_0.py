from flask import Flask, request
import lxml.etree

@app.route("/process", methods=['POST'])
def process_xml():
    ''' 
    Process the XML data from the request, convert it into a parse-able document,
    and then convert back and return as a string.
