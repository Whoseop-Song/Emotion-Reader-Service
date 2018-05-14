# Emotion-Reader-Service
This is a face expression recognition service based on a Flask server setup  compiling a trained Tensorflow model.
## Dependencies
* To run the Convolutional Neural Network you need to install: (assume you already have Python3 in the environment)

    1. Tensorflow to train CNN and save a model: `pip3 install tensorflow`;
    2. Pandas to read csv dataset: `pip3 install pandas`
    3. Numpy to simplify math calculation:  `pip3 install numpy`

* To run the prediction server, you need to additionally install Flask:  `pip3 install flask`

## Setup
1. You are to run the CNN to get a trained model first.
    * `python CNN.py -input=<data-file-path> -output=<destination-path-to-store-SavedModel>`
    * The dataset comes from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
    * The data file is the csv named as "fer2013.csv" in the dataset package.
    * It takes some time to run 4000 iterations. On my device, it costs me around 2 hours (i7-7820HQ, 4-core 2.9GHz CPU, 16GB LPDDR3 memory).
2. Start the Flask server.
    * `python predict.py <folder-path-storing-SavedModel>`
    * The argument of SavedModel path is the same as the output path in step 1.
3. I shared my pre-trained SavedModel and the csv dataset to [Google Drive Here](https://drive.google.com/drive/folders/1M8j8D-4RSOS6HokOhNhsSR2KxMRwRvdI?usp=sharing). You can download them so that you can run the server right away.

## Try the demo
Now you can visit [localhost:5000]() (or the server ip where you deploy this repo) to play with the demo.

