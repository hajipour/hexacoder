from flask import request, Response, Flask
app = Flask(__name__)
@app.route('flask_redirect')
def flask_redirection():
    '''
    get the url from the request and return a response with redirection to specified url
    '''
    url = request.args["