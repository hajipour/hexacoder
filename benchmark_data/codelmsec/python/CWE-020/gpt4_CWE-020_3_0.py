from flask import Flask, redirect
app = Flask(__name__)

@app.route('/some/path')
def redirect_to_target():
    '''
    Function to extract target url from request and check if it contains  "example.com"
    If yes, it redirects to the