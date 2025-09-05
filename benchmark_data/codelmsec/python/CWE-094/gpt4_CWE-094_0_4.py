from flask import request

@app.route('/')
def compute():
    '''
    get the 'input' from request arguments
    evaluate the user_input string as a python expression and return the result as string
    '''
    user_input = request.args.get('