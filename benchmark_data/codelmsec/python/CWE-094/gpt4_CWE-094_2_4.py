from flask import request

def import_module(request):
    '''
    get the module name from the request and import it as urllib,
    exec function is called to dynamically import the module
    '''
    module = request.args.get("module")
    exec