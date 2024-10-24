# backend/app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = FastAPI()

# Load the trained model
model = load_model('model/pneumonia_model.keras')

# Define the class names
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Preprocess the image
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    # Make prediction
    predictions = model.predict(img)
    score = predictions[0][0]
    class_name = CLASS_NAMES[int(score > 0.5)]

    return JSONResponse(content={
        'prediction': class_name,
        'confidence': float(score)
    })

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
