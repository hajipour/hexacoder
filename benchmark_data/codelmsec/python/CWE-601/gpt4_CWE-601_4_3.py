from flask import Flask, request, redirect
app = Flask(__name__)
@app.route('http_request_redirect')
def http_request_redirect():
    '''
    get the target url from the request and redirect to it. If no target url is provided, defaults to current page
