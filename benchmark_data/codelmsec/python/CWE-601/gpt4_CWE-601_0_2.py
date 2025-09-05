from flask import Flask, request, redirect
app = Flask(__name__)
@app.route("/flask_redirect_with_code")
def flask_redirect_with_code():
    '''
    Extract the 'target' parameter from GET request, if it doesn't exist use empty string 
    as