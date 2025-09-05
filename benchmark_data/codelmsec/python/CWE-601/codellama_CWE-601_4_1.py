from flask import Flask, request, redirect

app = Flask('flask_redirect_with_code')
@app.route('/')
def redirect_with_code():
    '''
        redirect with code 302
    '''