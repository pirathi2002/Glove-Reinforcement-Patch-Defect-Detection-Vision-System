import PySpin
import cv2
import numpy as np
from src.preprocessing import preprocess
from src.anomaly_model_traing import anomaly_score
from src.heatmap import create_heatmap
from src.classifier import final_decision
from src.logger import AnomalyLogger

def setup_hardware_trigger(cam):
    """
    Configure camera to respond to external hardware trigger.
    Trigger settings:
        - TriggerMode Off → configure → TriggerMode On
        - TriggerSelector FrameStart
        - TriggerSource Line0 (connected to sensor)
        - TriggerActivation RisingEdge
    """
    cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
    cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
    cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
    cam.TriggerMode.SetValue(PySpin.TriggerMode_On)

def split_two_gloves(image):
    """
    Assuming the image contains two gloves side by side,
    this function splits the image into two separate pre-processed images.
    """
    height, width = image.shape[:2]
    mid = width // 2
    glove1 = image[:, :mid]   # Left glove
    glove2 = image[:, mid:]   # Right glove
    return glove1, glove2

def run_real_time_two_gloves():
    """
    Main real-time loop:
        - Initialize FLIR camera
        - Wait for hardware trigger pulse
        - Capture image containing 2 gloves
        - Split image into 2 glove regions
        - Preprocess and run anomaly detection
        - Generate heatmaps
        - Log results
        - Display images for debugging
    """
    # Initialize camera
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError("No FLIR camera detected")

    cam = cam_list.GetByIndex(0)
    cam.Init()

    # Fixed exposure and gain for consistent lighting
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.ExposureTime.SetValue(7000)  # adjust for your lighting
    cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    cam.Gain.SetValue(0)

    setup_hardware_trigger(cam)
    cam.BeginAcquisition()

    # Initialize logger
    logger = AnomalyLogger()
    glove_counter = 0

    try:
        while True:
            glove_counter += 1
            print(f"\n[INFO] Waiting for image {glove_counter} (contains 2 gloves)...")

            # Capture image on hardware trigger (timeout 5s)
            image = cam.GetNextImage(5000)
            if image.IsIncomplete():
                print("[WARNING] Incomplete image. Skipping.")
                image.Release()
                continue

            img = image.GetNDArray()
            image.Release()

            # ----- DEBUG: show original captured image -----
            cv2.imshow("Captured Image (2 gloves)", img)

            # ----- SPLIT IMAGE INTO TWO GLOVES -----
            glove1_img, glove2_img = split_two_gloves(img)

            # Loop through each glove for preprocessing and analysis
            for i, glove_img in enumerate([glove1_img, glove2_img], start=1):
                print(f"[INFO] Processing glove {i} in image {glove_counter}...")

                # ----- PREPROCESSING -----
                preprocessed = preprocess(glove_img, resize_dim=(224, 224), debug=False)

                # ----- ANOMALY DETECTION -----
                score, recon = anomaly_score(preprocessed)
                heatmap = create_heatmap(preprocessed, recon)
                result = final_decision([score])

                # ----- LOGGING -----
                logger.log(
                    image_id=f"image_{glove_counter:03d}_glove_{i}",
                    preprocessed=preprocessed,
                    heatmap=heatmap,
                    score=score,
                    result=result
                )

                # ----- DEBUG: show PRE and Heatmap -----
                # Scale preprocessed image to 0-255 for display
                display_pre = (preprocessed * 255).astype(np.uint8)
                cv2.imshow(f"Preprocessed Glove {i}", display_pre)
                cv2.imshow(f"Heatmap Glove {i}", heatmap)

                # Print debug info
                print(f"[DEBUG] Glove {i} result: {result}, score: {score:.4f}")
                print(f"[DEBUG] Preprocessed shape: {preprocessed.shape}, Heatmap shape: {heatmap.shape}")

            # ----- EXIT CONDITION -----
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exiting real-time inspection.")
                break

    finally:
        # Cleanup camera and windows
        cam.EndAcquisition()
        cam.DeInit()
        cam_list.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()
