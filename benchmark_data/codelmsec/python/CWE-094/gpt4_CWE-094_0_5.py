from flask import request

@app.route('/')
def evaluate_input():
    '''
    Get 'input' from the request arguments.
    Run it as a Python expression and return the evaluated result as a string.
    '''
    user_input = request.args['input