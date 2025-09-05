from flask import request
import yaml


app = Flask(__name__)
@app.route('api/temperature_from_yaml')
def temperature_from_yaml():
    data = request
