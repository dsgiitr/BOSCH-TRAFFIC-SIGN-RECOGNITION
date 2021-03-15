import flask
from flask_cors import CORS, cross_origin
from flask import request, jsonify, send_file, redirect, url_for
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

# @cross_origin()
# @app.route("/SelectType", methods=["POST"])
# def getType():
#     json_data = json.loads(request.data)
#     type = json_data["data_selection"]
#     if type == "random":
#         return redirect("/RandomType")
#     elif type == "manual":
#         return redirect("/ManualType")
#     resp = jsonify(success=True)
#     return resp 

@cross_origin()
@app.route("/RandomType", methods=["POST"])
def getRandom():
    if request.data == None:
        pass
    else:
        json_data = json.loads(request.data)
        folder = json_data["type_of_data"]
        percent = int(json_data["percentage"])
        util.create_random_batch(folder, percent)
    resp = jsonify(success=True)
    return resp

@cross_origin()
@app.route("/ManualType", methods=["POST"])
def getManual():
    if request.data == None:
        pass
    else:
        json_data = request.data
        util.create_manual_batch(json_data)
        resp = jsonify(success=True)
    return resp 

if __name__ == '__main__':
    app.run(debug=True)
