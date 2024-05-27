import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def relU(Z):
    return np.maximum(Z, 0)

def relU_prime(Z):
    return Z > 0

def softMax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def get_predictions(A2):
    return np.argmax(A2, 0), np.max(A2)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions, confidence = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2, X, Y):
    current_image = X[:, index, None]
    prediction = make_predictions(X[:, index, None], W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def initialize_parameters():
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    return W1, b1, W2, b2


# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1
    A1 = relU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softMax(Z2)
    return Z1, A1, Z2, A2

# Backward propagation
def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2, m_var):
    dZ2 = A2 - one_hot(Y)
    dW2 = 1/m_var * dZ2.dot(A1.T)
    dB2 = 1/m_var * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * relU_prime(Z1)
    dW1 = 1/m_var * dZ1.dot(X.T)
    dB1 = 1/m_var * np.sum(dZ1)
    return dW1, dB1, dW2, dB2

# Update the weights
def update_weights(W1,W2,b1,b2,dW1,dB1,dW2,dB2,alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * dB1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * dB2
    return W1, W2, b1, b2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Gradient descent
def gradient_descent(X,Y,alpha,iterations,m_var):
    W1, b1, W2, b2 = initialize_parameters()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, dB1, dW2, dB2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2, m_var)
        W1, W2, b1, b2 = update_weights(W1, W2, b1, b2, dW1, dB1, dW2, dB2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions, confidence = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, W2, b1, b2
