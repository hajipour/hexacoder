from flask import request

@app.route('/')
def get_evaluated_input():
    '''
    getting 'input' parameter from the GET request and evaluate it as a python expression
    returning string representation of that result
    '''
    user_input = request.args