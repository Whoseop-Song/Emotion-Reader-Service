import tensorflow as tf
import pandas as pd
import numpy as np
from flask import Flask, request, Response
app = Flask(__name__)

def load_test_data(file_path):
    data = pd.read_csv(file_path)
    data = data[data.Usage == "PublicTest"]
    pixels = data.pixels.str.split(" ").tolist()
    pixels = pd.DataFrame(pixels, dtype=int)
    return pixels.values.astype(np.float), data["emotion"].values.ravel()

def pre_processing(pixels, labels):
    #standarize pixels
    pixels = pixels - pixels.mean(axis=1).reshape(-1, 1)
    pixels = np.multiply(pixels, 100.0/255.0)
    each_pixel_mean = pixels.mean(axis=0)
    each_pixel_std = np.std(pixels, axis=0)
    pixels = np.divide(np.subtract(pixels, each_pixel_mean), each_pixel_std)
    #change labels numbers to vectors
    label_vectors = np.zeros((labels.shape[0], 7))
    for i in range(labels.shape[0]):
        label_vectors[i][labels[i]] = 1
    return pixels, label_vectors.astype(np.uint8)

def load_predictor(model_path):
    predict = tf.contrib.predictor.from_saved_model(model_path)
    return predict

@app.route('/')
def main_page():
    return Response(open('./index.html').read(), mimetype="text/html")

if __name__=="__main__":
    predict = load_predictor('/Users/kalryoma/Downloads/model')
    pixels, labels = load_test_data("/Users/kalryoma/Downloads/fer2013/fer2013.csv")
    pixels, labels = pre_processing(pixels, labels)

    prediction = predict({"x": pixels, "y_": labels})
    prediction = np.argmax(prediction["y"], axis=1)

    app.run()
