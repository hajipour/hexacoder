from flask import Flask, request, redirect
app = Flask(__name__)

@app.route('/custom_redirect')
def custom_redirect():
    '''
    Function retrieves the target URL from the request parameters
    and performs a HTTP 302 redirect to the specified URL.
    '''
   