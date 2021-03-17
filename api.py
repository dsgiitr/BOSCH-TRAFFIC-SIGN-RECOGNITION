from flask import Flask, request, send_from_directory, request, jsonify, send_file, redirect, url_for
from flask_cors import CORS, cross_origin
import json
import utils.volatile as util
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__, static_url_path='')
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
@app.route("/SendSplit/<timestamp>", methods=["GET"])
def getSplitData(timestamp):
    json_file = util.create_train_test_json()
    return send_file(json_file)

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


@cross_origin()
@app.route("/GetOrg16/<timestamp>", methods=["GET"])
def getOriginal16(timestamp):
    json_file = util.create_org16_json()
    return send_file(json_file)


@cross_origin()
@app.route("/GetMod16/<timestamp>", methods=["GET"])
def getModified16(timestamp):
    json_file = util.create_mod16_json()
    return send_file(json_file)


@app.route('/data/<path:path>')
def send_js(path):
    return send_from_directory('data/', path)

@cross_origin()
@app.route("/SendTransform1", methods=["POST"])
def apply1():
    json_data = request.data
    util.apply_single(json_data)
    resp = jsonify(success=True)
    return resp

@cross_origin()
@app.route("/SendTransform16", methods=["POST"])
def apply16():
    json_data = request.data
    util.apply_16(json_data)
    resp = jsonify(success=True)
    return resp

@cross_origin()
@app.route("/SendTransformBatch", methods=["POST"])
def applybatch():
    util.apply_batch()
    resp = jsonify(success=True)
    return resp

if __name__ == '__main__':
    app.run(debug=True)