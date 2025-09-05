from flask import request

@app.route('/evaluate', methods=['GET'])
def evaluate_expression():
    '''
    Get 'input' from the query parameters,
    Use python eval function to run the input as a python expression.
    Return the result after evaluated