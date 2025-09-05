from flask import Flask, request, make_response

app = Flask(__name__)

@app.route('/response', methods =['GET'])
def get_response():
    '''
    Fetch data "name" from request
    Try to concatenate given data to string
    Make a response from