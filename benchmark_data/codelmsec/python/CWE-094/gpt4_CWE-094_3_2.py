from flask import request

@app.route('/')
def evaluate_expression():
    '''
    get the user_input from the request arguments
    evaluate it as python expression using eval function,
    lastly, return the evaluation result as string
    '''
    user_input