import os
import cv2
import json
import uuid
import requests
import numpy as np
from tensorflow.keras import models
from flask import Flask, request, jsonify

MODEL_PATH = "./tf-cnn-model.h5"

app = Flask(__name__)
app.model = models.load_model(MODEL_PATH)

@app.route("/")
def home():
	return "Hello, World!"


@app.route("/detect", methods=['POST'])
def detect():
	try:
		req = json.loads(request.get_data())
	except:
		return create_error_response("", "BAD_REQUEST")

	reference_id = req.get("reference_id")
	image_url = req.get("image_url")
	
	if image_url is None:
		return create_error_response(reference_id, "BAD_REQUEST")

	# download image
	try:
		r = requests.get(image_url, allow_redirects=True)
	except Exception as e:
		print("[ERROR] Downloading image: ", e, "; REFERENCE ID: ",reference_id)
		return create_error_response(reference_id, "BAD_REQUEST")
	
	uuid_string = str(uuid.uuid4())
	image_path = "tmp/" + uuid_string + ".jpg"
	open(image_path, 'wb').write(r.content)

	# predict digit 
	predicted_digit = predict_digit(image_path)
	response = create_success_response(reference_id, predicted_digit)

	# delete image
	os.remove(image_path)

	return response


def predict_digit(image_path):
	image = cv2.imread(image_path, 0)      
	image1 = cv2.resize(image, (28,28))
	image2 = image1.reshape(1,28,28,1)
	pred = np.argmax(app.model.predict(image2), axis=-1)
	return pred[0]    


def create_success_response(reference_id, predicted_digit):
	response = {}
	response["reference_id"] = reference_id
	response["predicted_digit"] = str(predicted_digit)
	response["status"] = "completed"
	return jsonify(response)


def create_error_response(reference_id, code):
	response = {}
	response["code"] = code 
	response["reference_id"] = reference_id
	response["status"] = "invalid_request"
	return jsonify(response)


if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True)