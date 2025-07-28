from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_ssim(imageA, imageB):
    # Resize to same dimensions if needed
    if imageA.shape != imageB.shape:
        imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

    score, diff = ssim(imageA, imageB, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff
