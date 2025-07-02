# photo_editor_ui.py (Düzeltilmiş ve Geliştirilmiş Versiyon)

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import math

# Import the core functionality
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

class PhotoEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Editor")
        self.root.geometry("600x600") # Log alanı için biraz daha yer açalım
        self.root.resizable(True, True)
        
        # Değişkenler
        self.input_folder = tk.StringVar(value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_photos"))
        self.output_folder = tk.StringVar(value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_photos"))
        self.width = tk.IntVar(value=83)
        self.height = tk.IntVar(value=109)
        self.processing = False
        
        # YOLO modelini yükle
        try:
            self.model = YOLO("yolov8m.pt")
            self.log("YOLO modeli başarıyla yüklendi!")
        except Exception as e:
            self.log(f"YOLO modeli yüklenirken hata: {e}")
            messagebox.showerror("Hata", f"YOLO modeli yüklenemedi: {e}")
            root.destroy()
            return
        
        # MediaPipe araçlarını yükle
        # Yüzdeki ince detayları bulmak için (hassas açı tespiti)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        
        # Fotoğrafın genel yönünü bulmak için (kaba açı tespiti)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
        
        self.create_widgets()
        
        # Gerekli klasörleri oluştur
        os.makedirs(self.input_folder.get(), exist_ok=True)
        os.makedirs(self.output_folder.get(), exist_ok=True)
        self.log("Uygulama başlatıldı. Klasörleri seçip ayarları yapabilirsiniz.")
        
    def create_widgets(self):
        # Ana Çerçeve
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Klasör seçiciler
        self.create_folder_selector(self.main_frame, "Giriş Klasörü:", self.input_folder, 0)
        self.create_folder_selector(self.main_frame, "Çıkış Klasörü:", self.output_folder, 1)
        
        # Boyut ayarları
        size_frame = ttk.LabelFrame(self.main_frame, text="Hedef Piksel", padding=10)
        size_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(size_frame, text="Genişlik:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(size_frame, from_=10, to=1000, textvariable=self.width, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(size_frame, text="Yükseklik:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(size_frame, from_=10, to=1000, textvariable=self.height, width=10).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # İşlem butonu
        self.process_button = ttk.Button(self.main_frame, text="Fotoğrafları İşle", command=self.start_processing)
        self.process_button.grid(row=3, column=0, columnspan=3, pady=20)
        
        # İlerleme çubuğu ve durum etiketi
        self.progress_bar = ttk.Progressbar(self.main_frame, orient=tk.HORIZONTAL, length=500, mode='determinate')
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)
        self.status_label = ttk.Label(self.main_frame, text="Hazır", anchor=tk.CENTER)
        self.status_label.grid(row=5, column=0, columnspan=3, sticky="ew")
        
        # Log alanı
        log_frame = ttk.LabelFrame(self.main_frame, text="İşlem Günlüğü", padding=10)
        log_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=10)
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(6, weight=1)
        
    def create_folder_selector(self, parent, label_text, folder_var, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=folder_var, width=50).grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(parent, text="Gözat...", command=lambda: self.browse_folder(folder_var)).grid(row=row, column=2, padx=5, pady=5)
    
    def browse_folder(self, folder_var):
        folder_path = filedialog.askdirectory(initialdir=folder_var.get())
        if folder_path:
            folder_var.set(folder_path)
            self.log(f"Seçilen klasör: {folder_path}")
    
    def log(self, message):
        if self.root.winfo_exists():
            self.root.after(0, self._log_update, message)
            
    def _log_update(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        print(message)
    
    def update_status(self, message):
        self.status_label.config(text=message)

    # --- YENİ VE GELİŞTİRİLMİŞ FONKSİYONLAR ---

    def get_face_orientation_angle(self, image):
        """
        Fotoğrafın ana yönünü (0, 90, 180, 270 derece) tespit eder.
        Bunu yapmak için görüntüyü her yöne çevirir ve en yüksek güven skoruna sahip yüzü bulur.
        """
        self.log("Fotoğrafın ana yönü tespit ediliyor...")
        best_angle = 0
        max_confidence = -1
        
        for angle in [0, 270, 180, 90]: # 270 ve 90'ı önce denemek genellikle daha iyi sonuç verir
            if angle > 0:
                rotated_img = self.rotate_image_by_90_degrees(image, angle)
            else:
                rotated_img = image
            
            # Yüz tespiti yap
            detection_results = self.face_detection.process(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
            
            if detection_results.detections:
                current_confidence = detection_results.detections[0].score[0]
                self.log(f"  -> {angle} derece açısı denendi. Yüz bulma güven skoru: {current_confidence:.2f}")
                if current_confidence > max_confidence:
                    max_confidence = current_confidence
                    best_angle = angle
        
        if max_confidence == -1:
            self.log("! Uyarı: Hiçbir açıda yüz bulunamadı. Döndürme yapılmayacak.")
            return 0
            
        self.log(f"-> En iyi açı {best_angle} derece olarak belirlendi.")
        return best_angle

    def get_precise_face_angle(self, image):
        """
        Gözleri referans alarak yüzün hassas eğim açısını (-15 ile +15 derece gibi) hesaplar.
        """
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return 0
        
        landmarks = results.multi_face_landmarks[0].landmark
        # Sol ve sağ gözün dış köşeleri (daha stabil sonuç verir)
        p_left = landmarks[33]  # Sol göz
        p_right = landmarks[263] # Sağ göz

        if p_left and p_right:
            dx = p_right.x - p_left.x
            dy = p_right.y - p_left.y
            angle = np.degrees(np.arctan2(dy, dx))
            return angle
        return 0

    def rotate_image_by_90_degrees(self, image, angle):
        """Görüntüyü 90, 180, 270 derece döndürür."""
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def rotate_image_with_correction(self, image, angle):
        """
        Görüntüyü hassas bir açıyla döndürür ve siyah kenarlıkları önler.
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Döndürme matrisini al
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Siyah kenarlıkları önlemek için yeni boyutları hesapla
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Matrisi yeni merkeze göre ayarla
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Görüntüyü döndür
        return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    def remove_black_borders(self, image):
        """
        Döndürme sonrası oluşan siyah kenar boşluklarını kırpar.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
            
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        return image[y:y+h, x:x+w]

    # --- ANA İŞLEM SÜRECİ ---

    def start_processing(self):
        if self.processing:
            return
        
        input_dir = self.input_folder.get()
        if not os.path.isdir(input_dir):
            messagebox.showerror("Hata", f"Giriş klasörü bulunamadı: {input_dir}")
            return
        
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not image_files:
            messagebox.showinfo("Bilgi", f"{input_dir} içinde hiç resim bulunamadı.")
            return
        
        # İşlem başlasın
        self.processing = True
        self.process_button.state(['disabled'])
        self.update_status("İşlem başlıyor...")
        self.progress_bar['maximum'] = len(image_files)
        self.progress_bar['value'] = 0
        
        threading.Thread(target=self.process_images_thread, args=(input_dir, self.output_folder.get(), (self.width.get(), self.height.get()), image_files), daemon=True).start()

    def process_images_thread(self, input_dir, output_dir, target_size, image_files):
        processed_count = 0
        success_count = 0
        
        for i, filename in enumerate(image_files):
            try:
                self.root.after(0, self.update_status, f"İşleniyor: {filename} ({i+1}/{len(image_files)})")
                
                path = os.path.join(input_dir, filename)
                image = cv2.imread(path)
                if image is None:
                    self.log(f"Hata: {filename} okunamadı.")
                    continue
                
                # ADIM 1: Fotoğrafın kaba yönünü düzelt (90, 180, 270 derece)
                coarse_angle = self.get_face_orientation_angle(image)
                if coarse_angle != 0:
                    self.log(f"-> {filename}: Ana yön {coarse_angle} derece düzeltiliyor...")
                    image = self.rotate_image_by_90_degrees(image, coarse_angle)
                
                # ADIM 2: Yüzün hassas eğimini düzelt
                precise_angle = self.get_precise_face_angle(image)
                if abs(precise_angle) > 1: # Sadece 1 dereceden büyük eğimleri düzelt
                    self.log(f"-> {filename}: Hassas eğim {precise_angle:.1f} derece düzeltiliyor...")
                    image = self.rotate_image_with_correction(image, precise_angle)
                    image = self.remove_black_borders(image) # Döndürme sonrası siyah köşeleri temizle
                else:
                    self.log(f"-> {filename}: Hassas eğim düzeltmeye gerek yok.")

                # ADIM 3: Kişiyi bul (YOLO)
                results = self.model(image, verbose=False) # `verbose=False` logları temiz tutar
                person_boxes = [box for box in results[0].boxes.data.cpu().numpy() if int(box[5]) == 0]
                
                if not person_boxes:
                    self.log(f"! Uyarı: {filename} içinde kişi bulunamadı. Tüm resim kullanılacak.")
                    cropped_image = image
                else:
                    # En büyük kişiyi al
                    biggest_person = max(person_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    x1, y1, x2, y2 = map(int, biggest_person[:4])
                    # Kişiyi kırp
                    cropped_image = image[y1:y2, x1:x2]
                    self.log(f"-> {filename}: Kişi bulundu ve kırpıldı.")
                
                # ADIM 4: Yeniden boyutlandır ve kaydet
                resized = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, resized)
                self.log(f"✅ {filename} başarıyla işlendi ve kaydedildi.")
                success_count += 1
                
            except Exception as e:
                self.log(f"❌ HATA: {filename} işlenirken bir sorun oluştu: {e}")
                # Hata durumunda orijinal resmi boyutlandırıp kaydetmeyi dene
                try:
                    original_image = cv2.imread(path)
                    resized = cv2.resize(original_image, target_size, interpolation=cv2.INTER_AREA)
                    save_path = os.path.join(output_dir, "FALLBACK_" + filename)
                    cv2.imwrite(save_path, resized)
                    self.log(f"-> Yedekleme: {filename} orijinal haliyle boyutlandırıldı.")
                except Exception as fallback_e:
                    self.log(f"-> Yedekleme de başarısız oldu: {fallback_e}")

            processed_count += 1
            self.root.after(0, lambda p=processed_count: self.progress_bar.config(value=p))
        
        # İşlem bitti
        self.root.after(0, self.update_status, f"İşlem tamamlandı. {success_count}/{processed_count} resim başarıyla işlendi.")
        self.root.after(0, lambda: self.process_button.state(['!disabled']))
        self.processing = False

if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoEditorApp(root)
    root.mainloop()
