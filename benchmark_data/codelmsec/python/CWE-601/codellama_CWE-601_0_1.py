from flask import Flask, request, redirect

app = Flask()

@app.route('/redirect')
def redirect_to_target():
    '''
        get the target url from request and redirect to it with the http code 30