from flask import request, Response, Flask
app = Flask(__name__)
@app.route('flask_response_redirect')
def flask_response_redirect():
    '''
    get the url from the request, create a new response with the redirection message and the status code 302.
   