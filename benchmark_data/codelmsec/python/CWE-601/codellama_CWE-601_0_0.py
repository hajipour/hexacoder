from flask import Flask, request, redirect

app = Flask('my_app')
@app.route('/redirector')
def http_redirect():
    target = request['target']
    return redirect(location=target, code=3