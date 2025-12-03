import cv2
import numpy as np
from typing import Tuple

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Process the uploaded image into MNIST format:
    - Load image
    - Convert to grayscale
    - Resize to 28x28
    - Normalize pixels
    - Reshape for model: (1, 28, 28, 1)
    """
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Invalid image or path")

    # Resize to 28x28
    image = cv2.resize(image, (28, 28))

    # Normalize pixel values
    image = image.astype("float32") / 255.0

    # Reshape for TF model
    image = np.reshape(image, (1, 28, 28, 1))

    return image
