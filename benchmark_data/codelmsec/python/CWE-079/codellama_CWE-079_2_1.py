from flask import Flask, request, make_response

app = Flask('app')

@app.get('/app/no_html_file_response')
def no_html_file_resposne():
    '''
        get the name data