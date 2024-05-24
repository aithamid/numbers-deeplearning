from typing import List, Union
from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
from functions import forward_propagation, get_predictions
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


model = np.load('model.npz')
W1 = model['W1']
W2 = model['W2']
b1 = model['b1']
b2 = model['b2']

model_cnn = keras.models.load_model('digit_recognizer.h5')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class ImageModel(BaseModel):
    image: List[Union[int, float]]

@app.post("/predict/")
async def predict(image: ImageModel):
    if len(image.image) != 784:
        return {"error": "Invalid image size"}
    # (784, 1)
    array = np.array(image.image)[:, None]
    _, _, _, A2 = forward_propagation(array, W1, b1, W2, b2)
    prediction, confidence  = get_predictions(A2)
    if len(prediction) != 1:
        return {"error": "Prediction error"}
    return {"guess": int(prediction), "confidence": float(confidence)}


@app.post("/predict_cnn/")
async def predict_cnn(image: ImageModel):
    if len(image.image) != 784:
        return {"error": "Invalid image size"}
    # (784, 1) values between 0 and 1

    
    array = np.array(image.image).reshape(-1,28,28,1)
    cnn_preds = model_cnn.predict(array)
    prediction = np.argmax(cnn_preds)
    confidence = float(np.max(cnn_preds))
    return {"guess": int(prediction), "confidence": confidence}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)