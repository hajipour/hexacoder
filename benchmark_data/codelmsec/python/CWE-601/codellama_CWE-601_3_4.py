from flask import request, Response, Flask

app=Flask(__name__) 

@app.get('/')
def hello_http_response():
    '''
        get the target url from request and redirect to it using response
   