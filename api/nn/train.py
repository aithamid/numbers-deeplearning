import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from functions import gradient_descent, test_prediction

# Load the data
data = pd.read_csv('../../dataset/train.csv')

# Reorder the data
data = np.array(data)
m, n = data.shape
#np.random.shuffle(data)

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

_, m_train = X_train.shape

# Train the model
W1, W2, b1, b2 = gradient_descent(X_train, Y_train, 0.10, 500, m)
test_prediction(0, W1, b1, W2, b2, X_train, Y_train)

# Save the model
np.savez('model.npz', W1=W1, W2=W2, b1=b1, b2=b2)

# dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
# print(get_accuracy(dev_predictions, Y_dev))





