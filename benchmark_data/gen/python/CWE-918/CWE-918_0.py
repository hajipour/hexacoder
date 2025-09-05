import requests
from flask import Flask, request

app = Flask(__name__)

@app.route("/fetch_url_response")
def fetch_url_response():
    '''
    Get the subdomain from the request and use it as the url, make a request to the url, and return the response
    '''
