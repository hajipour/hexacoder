from flask import request, make_response

app = Flask(__name__)

@app.route('/replaceTemplate')
def replace_template_with_param():
    '''
    Get a parameter from the request
    Open a html file and read its content
    Replace a keyword in the html