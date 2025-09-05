from flask import request

@app.route('/')
def retrieve_evaluated_input():
    '''
    get the user input from the request
    evaluate the user input as a python expression and 
    return it as a string output
    '''
    user_input =