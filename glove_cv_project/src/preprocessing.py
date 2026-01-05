# src/preprocessing.py
import cv2
import numpy as np
# want to ask about resize value
def preprocess(img, resize_dim=(224, 224), debug=False):
    """
    Preprocess a glove image for anomaly detection.

    Steps:
        1. Apply histogram equalization for contrast enhancement
        2. Resize to model input dimensions

    """

    # Step 1: Contrast enhancement (Histogram Equalization)
    equalized = cv2.equalizeHist(img)
    if debug:
        print(f"[DEBUG] After histogram equalization")
        cv2.imshow("Equalized", equalized)
        cv2.waitKey(1)

    # Step 4: (optional) Resize to model input dimensions
    resized = cv2.resize(equalized, resize_dim)
    if debug:
        print(f"[DEBUG] Resized shape: {resized.shape}")
        cv2.imshow("Resized", resized)
        cv2.waitKey(1)

    return resized



