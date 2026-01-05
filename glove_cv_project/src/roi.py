import cv2
from src.preprocessing import preprocess  # Import your preprocessing function

# -------- CONFIG --------
DEBUG = False
# Hardcoded crop coordinates (change manually to ROI)
x, y, w, h = 50, 50, 150, 150  # example values
# ------------------------

def crop_roi(resized_img):
    """
    Crops a fixed ROI from a resized image.

    Parameters:
        resized_img: np.array
            Image returned from src.preprocessing.py
    """
    # Crop ROI
    roi = resized_img[y:y+h, x:x+w]

    # Prepare debug image
    debug_img = resized_img.copy()
    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if DEBUG:
        cv2.imshow("Resized Image", debug_img)
        cv2.imshow("Cropped ROI", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return roi, debug_img



   