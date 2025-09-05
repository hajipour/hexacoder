from flask import Flask, request
import lxml.etree

import lxml.objectify

app = Flasket(__name__)
@app.post('/upload')
def upload():

    xml_src=request.get_data() 
    doc