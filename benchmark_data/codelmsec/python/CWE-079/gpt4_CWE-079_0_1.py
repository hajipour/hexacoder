from flask import request, make_response

@app.route('/replace')
def replace_name_on_response():
    '''
    get a name (param) from request
    open the shared.html file and read it to html
    replace placeholder {{name}} on html using '