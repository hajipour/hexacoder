from flask import Flask, request, make_response

app = Flask( __name__ )

@app.errorhandler(404)
def page_not_found(e):
    return make_reponse('<h1>Page not