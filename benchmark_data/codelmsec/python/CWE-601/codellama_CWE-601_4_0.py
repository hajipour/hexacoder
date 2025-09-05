from flask import Flask, request, redirect

app = Flask()
@app.route('/redirect')
def redirect_with_status_code():
    '''
    1. get the target url from request
    2. redirect to the url