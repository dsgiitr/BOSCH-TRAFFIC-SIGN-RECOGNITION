import flask
from flask import request, jsonify, send_file
import json
import utils.volatile as util

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route("/InitialData", methods=["GET"])
def getInitialData():
    json_file = util.create_original_json()
    return send_file(json_file)

@app.route("/GenerateDataset", methods=["POST"])
def generateDataset():
    json_file = request.files['modified_structure.json']
    util.transfer_to_modified(json_file)

@app.route("/SplitData", methods=["POST"])
def splitDataset():
    json_data = request.data
    util.transfer_to_split(json_data)

if __name__ == '__main__':
   app.run(debug = True)
