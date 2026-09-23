import cv2
import numpy as np
import tensorflow as tf
import json
import os

from src.models import IMG_SIZE, get_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prototxt_path = os.path.join(BASE_DIR, "models", "face_detector", "deploy.prototxt")
caffemodel_path = os.path.join(BASE_DIR, "models", "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

face_net = None
if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
    # Use readNet instead of readNetFromCaffe for better compatibility with headless builds
    face_net = cv2.dnn.readNet(caffemodel_path, prototxt_path)
else:
    print(f"Warning: Face detector model not found at {prototxt_path}")

# Color mappings (BGR for OpenCV)
COLOR_MAP = {
    "with_mask": (0, 255, 0),             # Green
    "without_mask": (0, 0, 255),          # Red
    "mask_weared_incorrect": (0, 165, 255)# Amber (Orange)
}

# Display Labels
LABEL_MAP = {
    "with_mask": "Correctly Worn",
    "without_mask": "No Mask",
    "mask_weared_incorrect": "Incorrectly Worn"
}

def load_inference_model(model_path, class_index_path):
    model_name = os.path.basename(model_path).replace(".h5", "")
    try:
        model = get_model(model_name)
        model.load_weights(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    with open(class_index_path, "r") as f:
        class_indices = json.load(f)
    
    # Reverse dict to get index -> class_name
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

def process_frame(frame, model, idx_to_class):
    """
    Detects faces, classifies them, and draws bounding boxes.
    Expects and returns a BGR frame (OpenCV format).
    """
    if face_net is None:
        cv2.putText(frame, "Error: Face detector not loaded!", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame
        
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
        
    face_net.setInput(blob)
    detections = face_net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections (lowered to 0.25 to catch faces occluded by masks)
        if confidence > 0.25:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure bounding box is within frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)
            
            # Calculate width and height
            face_w = endX - startX
            face_h = endY - startY
            
            if face_w < 10 or face_h < 10:
                continue
                
            # Crop the face
            face_crop = frame[startY:endY, startX:endX]
            
            # Preprocess for model — match Keras ImageDataGenerator (RGB, 0-255 float32)
            face_resized = cv2.resize(face_crop, IMG_SIZE)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)   # BGR → RGB
            face_norm = face_rgb.astype("float32")                      # model expects 0-255
            face_array = np.expand_dims(face_norm, axis=0)             # add batch dim
            
            # Predict
            preds = model.predict(face_array, verbose=0)[0]
            class_idx = np.argmax(preds)
            mask_confidence = preds[class_idx]
            
            # Skip very low-confidence detections
            if mask_confidence < 0.45:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (150, 150, 150), 2)
                cv2.putText(frame, f"Uncertain ({mask_confidence*100:.0f}%)", (startX, startY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
                continue
            
            class_name = idx_to_class[class_idx]
            
            # Get styling
            color = COLOR_MAP[class_name]
            label_text = f"{LABEL_MAP[class_name]} ({mask_confidence*100:.1f}%)"
            
            # Draw BBox
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            # Draw Label Background
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (startX, startY - text_h - 10), (startX + text_w, startY), color, -1)
            
            # Draw Text
            cv2.putText(frame, label_text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return frame
