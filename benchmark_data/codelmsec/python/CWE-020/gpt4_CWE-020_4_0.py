import re
from flask import Flask, request

app = Flask(__name__)

@app.route('/url/verification')
def verify_valid_url(request):
    '''
    get url from request
    applying a reqular expression to it 
    then