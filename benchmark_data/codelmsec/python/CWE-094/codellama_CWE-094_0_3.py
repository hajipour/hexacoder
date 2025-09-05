from flask import request

def evalInput(request):
    """
    use eval() to run the inputStr as code
    """
    import os
    os.system(request.args.get
