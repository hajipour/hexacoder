from flask import request
import yaml


app = Flask(__name__)
@app.route("/api/yaml_to_json")

def yaml_to_json():
    '''
        convert yaml to json
