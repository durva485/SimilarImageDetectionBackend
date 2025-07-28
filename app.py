from flask import Flask, request, jsonify
from flask_cors import CORS
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/compare": {"origins": "https://similar-image-detection.vercel.app"}})  # ← Updated

def read_and_resize(file):
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, (256, 256))

@app.route("/compare", methods=["POST"])
def compare_images():
    file1 = request.files.get("image1")
    file2 = request.files.get("image2")

    if not file1 or not file2:
        return jsonify({"error": "Missing image(s)"}), 400

    img1 = read_and_resize(file1)
    img2 = read_and_resize(file2)

    score, _ = ssim(img1, img2, full=True)
    return jsonify({"similarity": score})

if __name__ == "__main__":
    app.run(debug=True)
