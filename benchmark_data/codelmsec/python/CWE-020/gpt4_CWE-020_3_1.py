from flask import Flask, redirect
app = Flask(__name__)

@app.route('/handler')
def handle_redirection():
    '''
    get target URL from request. The second parameter in the get 
    method provides a default value if 'target'
    doesn't