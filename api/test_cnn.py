#imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plot
# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.models.load_model('digit_recognizer.h5')

# Load the data
test = pd.read_csv("../dataset/test.csv")
test1 = test.values.reshape(-1,28,28,1)/255
cnn_preds = model.predict(test1)
print(cnn_preds)
print(np.max(cnn_preds,axis=1))
print(np.argmax(cnn_preds,axis=1))
