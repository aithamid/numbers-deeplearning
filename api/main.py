from typing import List, Union
from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
from functions import forward_propagation, get_predictions
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


model = np.load('model.npz')
W1 = model['W1']
W2 = model['W2']
b1 = model['b1']
b2 = model['b2']

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
    print(image.image)
    if len(image.image) != 784:
        return {"error": "Invalid image size"}
    # (784, 1)
    array = np.array(image.image)[:, None]
    _, _, _, A2 = forward_propagation(array, W1, b1, W2, b2)
    print(A2)
    prediction, confidence  = get_predictions(A2)
    print(prediction)
    if len(prediction) != 1:
        return {"error": "Prediction error"}
    return {"guess": int(prediction), "confidence": float(confidence)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)