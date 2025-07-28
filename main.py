from flask import Flask, request, jsonify
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from flask_cors import CORS 

app = Flask(__name__)

CORS(app, origins=["https://similar-image-detection.vercel.app"])


def read_and_resize(file):
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, (256, 256))

@app.route("/compare", methods=["POST"])
def compare_images():
    image1 = request.files.get("image1")
    image2 = request.files.get("image2")

    if not image1 or not image2:
        return jsonify({"error": "Missing image(s)"}), 400

    img1 = read_and_resize(image1)
    img2 = read_and_resize(image2)

    score, _ = ssim(img1, img2, full=True)
    return jsonify({"similarity": score})

if __name__ == "__main__":
    app.run(debug=True)
