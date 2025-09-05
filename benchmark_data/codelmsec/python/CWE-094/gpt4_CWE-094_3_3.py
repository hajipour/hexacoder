from flask import request

@app.route('/')
def evaluate_input():
    '''
    The function gets input from user and evaluates it as a python expression.
    'input' parameter in the request is used as a python expression.
    The result of this expression