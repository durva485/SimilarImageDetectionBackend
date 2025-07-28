from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

app = FastAPI()

# CORS for local + deployment
origins = [
    "http://localhost:3000",
    "https://your-frontend.vercel.app",
    "https://your-frontend.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],  # ← important for Swagger & preflight
    allow_headers=["*"],
)

def read_and_resize(file_data: bytes):
    img_array = np.frombuffer(file_data, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, (256, 256))

@app.post("/compare")
async def compare_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    if not image1 or not image2:
        raise HTTPException(status_code=400, detail="Missing image(s)")

    try:
        img1_data = await image1.read()
        img2_data = await image2.read()

        img1 = read_and_resize(img1_data)
        img2 = read_and_resize(img2_data)

        score, _ = ssim(img1, img2, full=True)
        return JSONResponse(content={"similarity": float(score)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
