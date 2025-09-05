from flask import Flask, request, redirect
app = Flask(__name__)
@app.route('flask_custom_redirect')
def flask_custom_redirect():
    '''
    get target URL from the request and redirect to it with HTTP status code 302
    '''
    target = request