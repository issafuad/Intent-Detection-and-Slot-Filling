__author__ = 'fuadissa'

import traceback
import logging
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

from atis_classifer.classifier import Classifier
from settings import MODEL_SERVING_DIR

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(level=logging.INFO)
LOGGER.addHandler(ch)

HOST = '0.0.0.0'
PORT = 5000
REQUEST_METHOD = 'classify'


APP = Flask(__name__)
APP_NAME = [key for key, val in locals().items() if val is APP][0]

MODEL = None

def load():
    global MODEL
    MODEL = Classifier(MODEL_SERVING_DIR)

@APP.route("/{}".format(REQUEST_METHOD), methods=["POST"])
def classify():
    try:
        json_request = request.get_json()
    except BadRequest:
        trace_back = traceback.format_exc()
        raise Exception("JSON is malformed.\n\n{}".format(trace_back))
    except Exception:
        raise Exception(traceback.format_exc())

    if not MODEL:
        load()

    results = MODEL.run_classifier(json_request['text'])

    return jsonify(results)

if __name__ == '__main__':
    APP.run(host=HOST, port=PORT)
