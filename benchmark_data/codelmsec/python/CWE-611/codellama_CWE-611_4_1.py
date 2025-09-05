from flask import Flask, request
import lxml.etree

from lxml.etree import XMLParser
import re

app = Flaska(__name__)
@app.post('/upload')
def upload():
#    xml = request.get_json()