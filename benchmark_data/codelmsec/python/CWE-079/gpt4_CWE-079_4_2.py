from flask import Flask, request, make_response

@app.route('/give_name_response', methods=['GET'])
def give_name_response():
    '''
    get the name parameter from request
    make a response by appending it with custom text
    return the response
    '''