import re
from flask import Flask

app = Flask(__name__)

@app.route('/api/check_url')
def validate_url(request):
    '''
    get url parameter from request
    use regular expression to check if it is a valid url 
