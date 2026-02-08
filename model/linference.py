"""
inference.py

Inference workflow for wafer defect classification.
Assumes a trained model will be available in later phases.
"""

import torch
import cv2
import numpy as np
from model_architecture import WaferDefectCNN


def preprocess_image(image_path):
    """
    Load and preprocess a grayscale image for inference.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0

    img = np.expand_dims(img, axis=0)  # Channel
    img = np.expand_dims(img, axis=0)  # Batch

    return torch.tensor(img, dtype=torch.float32)


def infer(image_path):
    classes = ["clean", "bridge", "open", "crack", "ler", "other"]

    model = WaferDefectCNN(num_classes=len(classes))
    model.eval()

    print("Inference pipeline initialized.")
    print("Model weights will be loaded in Phase-2.")

    image = preprocess_image(image_path)

    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    print("Predicted defect type:", classes[predicted_class])


if __name__ == "__main__":
    infer("sample_image.png")
