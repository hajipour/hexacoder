from flask import Flask, request, make_response

app = Flask(__name__)

@app.route('/get_name', methods =['GET'])
def get_name():
    '''
    capture the name data from request object 
    consize a response string from captured data 'Your name is