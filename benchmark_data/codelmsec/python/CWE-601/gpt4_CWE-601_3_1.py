from flask import request, Response, Flask
app = Flask(__name__)
@app.route('flask_http_response')
def flask_http_response():
    '''
    get the url from the request and generate a HTTP response with 302 status code 
    for directing the server to