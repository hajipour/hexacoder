from flask import Flask, redirect
app = Flask(__name__)

@app.route('/redirect')
def target_redirection():
    '''
    Fetch 'target' from request arguments
    Check if "example.com" is part of the 'target'
    If condition is