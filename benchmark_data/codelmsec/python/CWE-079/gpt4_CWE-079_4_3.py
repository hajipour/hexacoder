from flask import Flask, request, make_response

@app.route('/get_name', methods =['GET'])
def return_name_as_str():
    '''
    get the name data from request
    make and return a response by concatenating the 'Your name is '::string with the