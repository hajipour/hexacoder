from flask import Flask, request, make_response

@app.route('/get_and_respond_name', methods=['GET'])
def get_and_respond_name():
    '''
    Get the 'name' parameter from request
    Create a response with the value of 'name'
    Return