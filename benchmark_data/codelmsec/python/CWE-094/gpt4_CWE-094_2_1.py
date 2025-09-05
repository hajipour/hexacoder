from flask import request

def eval_module_import(request):
    '''
    get the module name from the request and import it dynamically using exec function
    '''
    module = request.args.get("module")
    exec("import urllib%s as urllib" %