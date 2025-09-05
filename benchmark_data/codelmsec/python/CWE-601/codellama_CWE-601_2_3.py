from flask import request, redirect, Flask

app=Flask(__name__)  
@app.route("redirect")
def redirection():
    '''
     get the url data from the request
     and redirect to it.
    '''