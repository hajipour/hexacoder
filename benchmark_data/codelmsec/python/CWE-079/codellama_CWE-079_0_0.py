from flask import request, make_response

import os

@app.get('/render')
def render_from_string():
    '''
     get param from request
    get shared.html from the templates
    replace {{ name }} with param