import os
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Giriş ve çıkış klasörlerini ayarla
input_folder = r"C:\Users\Dersu\Desktop\Deneme proje\Foto Deneme"
output_folder = r"C:\Users\Dersu\Desktop\Deneme proje\Foto Cikti"
target_size = (83, 109)

os.makedirs(output_folder, exist_ok=True)

# YOLOv8 modelini yükle (aynı dizinde yolov8m.pt dosyası olmalı)
model = YOLO("yolov8m.pt")

# MediaPipe yüz mesh başlat
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

def get_face_angle(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return 0
    face_landmarks = results.multi_face_landmarks[0].landmark
    left_eye = face_landmarks[33]  # sol göz
    right_eye = face_landmarks[263]  # sağ göz
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)

# Görselleri işle
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(input_folder, filename)
        image = cv2.imread(path)
        if image is None:
            print(f"Dosya okunamadı: {filename}")
            continue

        # Yüz açısını düzelt
        face_angle = get_face_angle(image)
        rotated_image = rotate_image(image, -face_angle)

        # Kişiyi yeniden algıla
        results = model(rotated_image)
        boxes = results[0].boxes.data.cpu().numpy()
        person_boxes = [box for box in boxes if int(box[5]) == 0]

        if not person_boxes:
            print(f"Kişi bulunamadı: {filename}")
            continue

        biggest = max(person_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = map(int, biggest[:4])
        person_crop = rotated_image[y1:y2, x1:x2]

        # Görseli yeniden boyutlandır
        resized = cv2.resize(person_crop, target_size, interpolation=cv2.INTER_AREA)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, resized)
        print(f"{filename} işlendi ✅")
