import flask
from flask_cors import CORS, cross_origin
from flask import request, jsonify, send_file
import json
import utils.volatile as util

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@cross_origin()
@app.route("/InitialData", methods=["GET"])
def getInitialData():
    json_file = util.create_original_json()
    return send_file(json_file)


@cross_origin()
@app.route("/GenerateDataset", methods=["POST"])
def generateDataset():
    json_data = request.data
    util.transfer_to_modified(json_data)
    resp = jsonify(success=True)
    return resp


@cross_origin()
@app.route("/SplitData", methods=["POST"])
def splitDataset():
    json_data = request.data
    util.transfer_to_split(json_data)
    resp = jsonify(success=True)
    return resp

@cross_origin()
@app.route("/SendSplit", methods=["GET"])
def getSplitData():
    json_dict = util.create_train_test_json()
    return jsonify(json_dict)

if __name__ == '__main__':
    app.run(debug=True)
