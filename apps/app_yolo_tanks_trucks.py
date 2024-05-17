from datetime import datetime
from flask import Flask, request, jsonify
from flask_api import status
from flask_cors import CORS
import utils_yolo
from utils_yolow import load_yolow, writeLog, detect_yolow

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = True
yolo_model = None

@app.route('/detect_yolo_tanks_trucks', methods=['POST'])
def detect_yolo():
    """
    POST
    Params:
    - imageData
    - confidence
    """
    body = request.get_json(force=True)
    if 'imageData' not in body:
        return jsonify({'response': 'imageData required'}), status.HTTP_400_BAD_REQUEST
    
    if 'confidence' not in body:
        confidence = 0.001
    else:
        confidence = body["confidence"]
    
    b64_image = body["imageData"]

    global yolo_model
    result = utils_yolo.detect_yolo(yolo_model, b64_image, confidence)

    result["status"] = "success"
    if "filter" in body:
        result = filter_results(result, body["filter"])

    return jsonify(result)

def filter_results(result, filter_cond):
    # Filter the objects based on the tag name
    filtered_objects = [obj for obj in result['objects'] if obj['tagName'] in filter_cond['classes'] and obj['confidence'] >= filter_cond['confidence']]
    result['objects'] = filtered_objects
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
    global yolo_model
    yolo_model = utils_yolo.load_yolo("best_tanks_militaryTrucks.pt")
    app.run(host='0.0.0.0', port=2055, debug=True, ssl_context=('cert/cert.pem', 'cert/ck.pem'))

if __name__ == "__main__":
    app_init()
