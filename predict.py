import tensorflow as tf
import pandas as pd
import numpy as np
from flask import Flask, request, Response, jsonify
app = Flask(__name__)
model_path = '/Users/kalryoma/Downloads/model'
predict = tf.contrib.predictor.from_saved_model(model_path)

def pre_processing(pixels):
    #standarize pixels
    # pixels = pixels - pixels.mean(axis=1).reshape(-1, 1)
    # pixels = np.multiply(pixels, 100.0/255.0)
    # each_pixel_mean = pixels.mean(axis=0)
    # each_pixel_std = np.std(pixels, axis=0)
    # pixels = np.divide(np.subtract(pixels, each_pixel_mean), each_pixel_std)
    #change labels numbers to vectors
    label_vectors = np.zeros((pixels.shape[0], 7))
    for i in range(pixels.shape[0]):
        label_vectors[i][0] = 1
    return pixels, label_vectors.astype(np.uint8)

@app.route('/')
def main_page():
    return Response(open('index.html').read(), mimetype="text/html")

@app.route('/face', methods=['POST'])
def get_face():
    global predict
    face_data = np.array(request.json)
    pixels, labels = pre_processing(face_data)
    prediction = predict({"x": pixels, "y_": labels})["y"]
    # prediction = np.argmax(prediction, axis=1)
    return jsonify(prediction.tolist())

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response

if __name__=="__main__":
    app.run(debug=True)
