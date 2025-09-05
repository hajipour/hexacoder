from flask import request

@app.route('/evaluate')
def handle_user_input():
    '''
    grab the user_input from request parameters
    evalute the user_input as Python code and return the result as string
    '''
    user_input = request.args