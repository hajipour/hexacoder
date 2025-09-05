from flask import Flask, request
import lxml.etree

app = Flask("flask")
@app.post('/')
def index():
    xml_src= request.get_data(as_text=True)
    doc = ltml.et