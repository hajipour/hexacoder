from flask import request, Response, Flask

app=Flask(__name__)  
@app.route("/")
def flask_response():
    '''
     get the url from the get request and redirect to it 
    '''
