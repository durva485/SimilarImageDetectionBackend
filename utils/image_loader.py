import cv2
import os

def load_and_prepare_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".bmp"]:
        raise ValueError(f"Unsupported file format: {ext}")

    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray
