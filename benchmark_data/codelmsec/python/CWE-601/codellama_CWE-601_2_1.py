from flask import request, redirect, Flask

app=Flask(__name__)    
@app.route("redirect_url")
def redirect_url():
    '''
     get url data from the requst and redirect to that url
   