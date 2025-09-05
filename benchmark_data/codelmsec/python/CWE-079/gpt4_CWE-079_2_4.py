from flask import Flask, request, make_response

app = Flask(__name__)

@app.route('/returnName', methods=['GET'])
def reponse_with_name():
    '''
    get the name from request
    create a response with string 'Your name is' appended with name