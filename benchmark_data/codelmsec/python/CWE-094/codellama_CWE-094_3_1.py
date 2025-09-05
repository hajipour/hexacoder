from flask import request

@app.route("/")
def index():

    '''
    getting user input from query string
    evaluating the user input and returning the result
    '''