from datetime import datetime
from flask import Flask, request, jsonify
from flask_api import status
from flask_cors import CORS
import utils
from utils import detect_orcnn, load_orcnn, writeLog

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = True
orcnn_model = None

@app.route('/detect_orcnn', methods=['POST'])
def detect_orcnn():
    """
    POST
    Params:
    - imageData
    - confidence
    """
    body = request.get_json(force=True)
    if 'imageData' not in body:
        return jsonify({'response': 'imageData required'}), status.HTTP_400_BAD_REQUEST

    
    b64_image = body["imageData"]

    global orcnn_model
    result = utils.detect_orcnn(orcnn_model, b64_image, 0.1)

    result["status"] = "success"
    if "filter" in body:
        result = filter_result(result, filter)

    return jsonify(result)


def filter_results(result, filter):
    # Filter the objects based on the tag name
    filtered_objects = [obj for obj in results['objects'] if obj['tagName'] in filter['classes'] and obj['confidence'] >= filter['confidence']]
    results['objects'] = filtered_objects
    return result


def formatResponse(response, list_field, page, limit):
    frmt_response = {}
    if page is not None and limit is not None:
        start = limit * (page - 1)
        end = limit * page
        frmt_response[list_field] = response[list_field][start:end]
        frmt_response["page"] = page
        frmt_response["limit"] = limit
        frmt_response["totalItems"] = len(response[list_field])
    else:
        frmt_response[list_field] = response[list_field]
        if isinstance(response[list_field], list):
            frmt_response["totalItems"] = len(response[list_field])

    if(response["status"] == "success"):
        r_status = status.HTTP_200_OK
    else:
        r_status = status.HTTP_400_BAD_REQUEST

    return jsonify(frmt_response), r_status


def app_init():
    global orcnn_model
    orcnn_model = load_orcnn()
    app.run(host='0.0.0.0', port=2053, debug=True, ssl_context=('cert/cert.pem', 'cert/ck.pem'))

if __name__ == "__main__":
    app_init()
