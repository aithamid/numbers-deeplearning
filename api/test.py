import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from functions import make_predictions, get_accuracy

# Load the data
data = pd.read_csv('../dataset/train.csv')

# Reorder the data
data = np.array(data)
m, n = data.shape
#np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

X_dev = X_dev / 255.

_, m_dev = X_dev.shape

# Load the model
model = np.load('model.npz')
W1 = model['W1']
W2 = model['W2']
b1 = model['b1']
b2 = model['b2']


print(X_dev.shape) # 784, 1000
# just 784, 1
print(X_dev[:, 0, None].shape)


# Make predictions
# create a test image zero dimension 0
# imageEmpty = np.zeros((784))
# # dim of 1, 784
# imageArray = imageEmpty / 255
# one_prediction = make_predictions(imageArray, W1, b1, W2, b2)

# print(one_prediction)

# print("next")

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print(dev_predictions)

print("next")
prediction = get_accuracy(dev_predictions, Y_dev)
print(prediction)