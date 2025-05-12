import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import torch

# Patch for PyTorch compatibility issues
if hasattr(torch, "serialization"):
    import importlib
    sys.modules["ultralytics.nn.tasks"] = importlib.import_module("ultralytics.nn.tasks")
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(["ultralytics.nn.tasks.DetectionModel"])

# Import YOLO after ensuring patches are in place
from ultralytics import YOLO

# Set input and output folders (using relative paths)
input_folder = "input_photos"
output_folder = "output_photos"
target_size = (83, 109)

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(input_folder, exist_ok=True)

# Initialize MediaPipe face mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

# Initialize YOLO model
try:
    model = YOLO("yolov8m.pt")
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    raise RuntimeError("Failed to load YOLO model. Please check your installation.")

def get_face_angle(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return 0
    face_landmarks = results.multi_face_landmarks[0].landmark
    left_eye = face_landmarks[33]  # left eye
    right_eye = face_landmarks[263]  # right eye
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)

# Process images
if not os.listdir(input_folder):
    print(f"Please place your images in the '{input_folder}' directory")
else:
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(input_folder, filename)
            image = cv2.imread(path)
            if image is None:
                print(f"Could not read file: {filename}")
                continue

            try:
                # Correct face angle
                face_angle = get_face_angle(image)
                rotated_image = rotate_image(image, -face_angle)
                print(f"Face angle detected and corrected: {face_angle:.1f} degrees")

                # Detect person using YOLO
                results = model(rotated_image)
                boxes = results[0].boxes.data.cpu().numpy()
                person_boxes = [box for box in boxes if int(box[5]) == 0]  # Class 0 is person in COCO dataset

                if not person_boxes:
                    print(f"No person detected in: {filename}, using the entire image")
                    # If no person detected, use the entire image
                    h, w, _ = rotated_image.shape
                    x1, y1, x2, y2 = 0, 0, w, h
                else:
                    # Find the largest person bounding box
                    biggest = max(person_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    x1, y1, x2, y2 = map(int, biggest[:4])
                
                # Crop to person
                person_crop = rotated_image[y1:y2, x1:x2]

                # Resize image
                resized = cv2.resize(person_crop, target_size, interpolation=cv2.INTER_AREA)
                save_path = os.path.join(output_folder, filename)
                cv2.imwrite(save_path, resized)
                print(f"Processed {filename} ✅")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                # Fallback: just resize the original image if processing fails
                try:
                    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                    save_path = os.path.join(output_folder, filename)
                    cv2.imwrite(save_path, resized)
                    print(f"Fallback: Resized {filename} without processing ✅")
                except Exception as fallback_error:
                    print(f"Fallback also failed: {str(fallback_error)}")

    print("\nAll images have been processed.")
    print(f"Processed images saved to: {os.path.abspath(output_folder)}")
