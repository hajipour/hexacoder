from flask import Flask, request, make_response

app = Flask( __name__ )

@app.get('/hello')
def hello():
    '''
    1. get the name parameter from request
    2. construct the response
    3