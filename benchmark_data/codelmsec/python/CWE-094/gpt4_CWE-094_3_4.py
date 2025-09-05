from flask import request

@app.route('/')
def evaluate_expression():
    '''
    Grab user input from the URL arguments 'input'.
    Use eval() to run this text as a Python expression.
    Wrap result in str and return.
    '''
    user