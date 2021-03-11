import flask
from flask import request, jsonify
import json
from utils.volatile import *

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route("/api/generate", methods=["POST"])
def generateDataset():
    json_data = json.loads(request.data)
    transfer_to_modified(json_data)
    return jsonify("success")


@app.route("/", methods=["GET"])
def ping():
    return jsonify("ping")


app.run()

