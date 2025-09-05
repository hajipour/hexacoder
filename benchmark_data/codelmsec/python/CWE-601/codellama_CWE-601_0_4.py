from flask import Flask, request, redirect

app = Flask()
@app.route('/')
def hello_world():
    '''
        get the url from the query string in the request,
        and redirect to the url
    '''