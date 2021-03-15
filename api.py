import flask
from flask_cors import CORS, cross_origin
from flask import request, jsonify, send_file
import json
import utils.volatile as util

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/InitialData", methods=["GET"])
@cross_origin()  # allow all origins all methods.
def getInitialData():
    json_file = util.create_original_json()
    return send_file(json_file)


@cross_origin()
@app.route("/GenerateDataset", methods=["POST"])
def generateDataset():
    dataset_json = json.loads(request.data)
    util.transfer_to_modified(dataset_json)
    resp = jsonify(success=True)
    return resp


@cross_origin()
@app.route("/SplitData", methods=["POST"])
def splitDataset():
    json_data = request.data
    app.logger.info(json_data)
    util.transfer_to_split(json_data)
    resp = jsonify(success=True)
    return resp


if __name__ == '__main__':
    app.run(debug=True)
