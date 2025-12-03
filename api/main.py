from fastapi import FastAPI, UploadFile, File
import numpy as np
from tensorflow import keras
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Load model once
model = keras.models.load_model("model/mnist_model.keras")

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))
    img_arr = np.array(image) / 255.0
    img_arr = img_arr.reshape(1, 28, 28, 1)
    return img_arr

@app.get("/")
def home():
    return {"message": "MNIST Digit Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img = preprocess(content)
    pred = model.predict(img)
    digit = int(np.argmax(pred))
    confidence = float(np.max(pred))
    return {"digit": digit, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
