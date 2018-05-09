import tensorflow as tf
import pandas as pd
import numpy as np

def load_test_data(file_path):
    data = pd.read_csv(file_path)
    data = data[data.Usage == "PublicTest"]
    pixels = data.pixels.str.split(" ").tolist()
    pixels = pd.DataFrame(pixels, dtype=int)
    return pixels.values.astype(np.float), data["emotion"].values.ravel()

if __name__=="__main__":
    pixels, labels = load_test_data("/Users/kalryoma/Downloads/fer2013/test.csv")
    pixels = np.multiply(pixels, 100.0/255.0)
    label_vectors = np.zeros((labels.shape[0], 7))
    label_vectors[0][0] = 1
    labels = label_vectors.astype(np.uint8)
    model_path = '/Users/kalryoma/Downloads/model'
    predict = tf.contrib.predictor.from_saved_model(model_path)
    prediction = predict({"x": pixels, "y_": labels})
    prediction = np.argmax(prediction["y"][0])
    print(prediction)
