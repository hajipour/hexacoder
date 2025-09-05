from flask import request, Response, Flask
app = Flask(__name__)
@app.route('/flask_redirect_302')
def flask_redirect_response():
    '''
    get the url from the request
    create a new response with the url in the headers to redirect
   