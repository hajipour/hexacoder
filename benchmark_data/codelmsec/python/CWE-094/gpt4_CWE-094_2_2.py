from flask import request

def import_module(request):
    '''
    get the module from the request args and import it using the exec method
    module specified is appended to urllib to import specific urllib module
    '''
    module = request.args.get("