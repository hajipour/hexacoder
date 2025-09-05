from flask import Flask, request, redirect
app = Flask(__name__)
@app.route('detect_target_redirect')
def detect_target_redirect():
    '''
    gets the target url from the request argument and redirects to it using code 302
    '''
    target = request.args