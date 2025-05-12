import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import math

# Import the core functionality from Duzenleyici
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

class PhotoEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Editor")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Initialize variables
        self.input_folder = tk.StringVar(value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_photos"))
        self.output_folder = tk.StringVar(value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_photos"))
        self.width = tk.IntVar(value=83)
        self.height = tk.IntVar(value=109)
        self.processing = False
        
        # Initialize YOLO model
        try:
            self.model = YOLO("yolov8m.pt")
            print("YOLO model loaded successfully!")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
            root.destroy()
            return
        
        # Initialize MediaPipe face mesh
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)
        
        # Initialize MediaPipe face detection for more reliable face orientation detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input folder selection
        self.create_folder_selector(self.main_frame, "Input Folder:", self.input_folder, 0)
        
        # Output folder selection
        self.create_folder_selector(self.main_frame, "Output Folder:", self.output_folder, 1)
        
        # Target size adjustment
        size_frame = ttk.LabelFrame(self.main_frame, text="Target Size", padding=10)
        size_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)
        
        # Width
        ttk.Label(size_frame, text="Width:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        width_spinbox = ttk.Spinbox(size_frame, from_=10, to=1000, textvariable=self.width, width=10)
        width_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Height
        ttk.Label(size_frame, text="Height:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        height_spinbox = ttk.Spinbox(size_frame, from_=10, to=1000, textvariable=self.height, width=10)
        height_spinbox.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Process button
        self.process_button = ttk.Button(self.main_frame, text="Process Photos", command=self.process_photos)
        self.process_button.grid(row=3, column=0, columnspan=3, pady=20)
        
        # Progress bar and status
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=500, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(self.progress_frame, text="Ready", anchor=tk.CENTER)
        self.status_label.pack(fill=tk.X)
        
        # Log area
        log_frame = ttk.LabelFrame(self.main_frame, text="Log", padding=10)
        log_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=10)
        
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid to expand
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(5, weight=1)
        
        # Create necessary folders
        os.makedirs(self.input_folder.get(), exist_ok=True)
        os.makedirs(self.output_folder.get(), exist_ok=True)
        
        # Add initial log
        self.log("Application started. Please select folders and adjust settings.")
        
    def create_folder_selector(self, parent, label_text, folder_var, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=5)
        
        folder_entry = ttk.Entry(parent, textvariable=folder_var, width=50)
        folder_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        
        browse_button = ttk.Button(parent, text="Browse...", 
                                command=lambda: self.browse_folder(folder_var))
        browse_button.grid(row=row, column=2, padx=5, pady=5)
    
    def browse_folder(self, folder_var):
        folder_path = filedialog.askdirectory(initialdir=folder_var.get())
        if folder_path:
            folder_var.set(folder_path)
            self.log(f"Selected folder: {folder_path}")
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        print(message)
    
    def update_status(self, message):
        self.status_label.config(text=message)
        self.log(message)

    def get_face_orientation_angle(self, image):
        """
        Detect the face orientation and return the angle needed to correct it.
        This handles arbitrary rotation angles, not just 90-degree increments.
        """
        # Try multiple orientations to find the best one
        test_angles = [0, 90, 180, 270, 45, 135, 225, 315]
        best_angle = 0
        max_score = -1
        face_found = False
        
        img_height, img_width = image.shape[:2]
        
        # First pass with standard angles
        for angle in test_angles:
            # Rotate image for testing
            if angle > 0:
                if angle in [90, 180, 270]:
                    # For 90-degree angles, use cv2.rotate which is lossless
                    rotated = self.rotate_image_by_angle(image, angle)
                else:
                    # For other angles, use the more complex rotation
                    rotated = self.rotate_image(image, angle)
            else:
                rotated = image.copy()
            
            # Try face detection on this orientation
            rgb_image = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            detection_results = self.face_detection.process(rgb_image)
            
            if detection_results.detections:
                face_found = True
                # Get highest confidence detection
                highest_conf = max(detection_results.detections, key=lambda det: det.score[0])
                
                # Calculate a score based on detection confidence and face position
                score = highest_conf.score[0] * 10
                
                # Get face bounding box
                bbox = highest_conf.location_data.relative_bounding_box
                face_center_y = bbox.ymin + bbox.height/2
                
                # Reward faces that are in upper half and centered horizontally
                if face_center_y < 0.6:
                    score += 3
                
                face_center_x = bbox.xmin + bbox.width/2
                horizontal_center_offset = abs(0.5 - face_center_x)
                if horizontal_center_offset < 0.2:
                    score += 2
                
                # Check if face is aligned properly (both eyes at same height)
                try:
                    if highest_conf.score[0] > 0.7:
                        mesh_results = self.face_mesh.process(rgb_image)
                        if mesh_results and mesh_results.multi_face_landmarks:
                            landmarks = mesh_results.multi_face_landmarks[0].landmark
                            
                            # Check eye alignment
                            left_eye = landmarks[33]   # Left eye
                            right_eye = landmarks[263] # Right eye
                            
                            # Eyes should be at same height (horizontally aligned)
                            if abs(left_eye.y - right_eye.y) < 0.03:
                                score += 5
                            
                            # Eyes should be relatively horizontal
                            eye_angle = np.degrees(np.arctan2(
                                right_eye.y - left_eye.y, 
                                right_eye.x - left_eye.x
                            ))
                            
                            if abs(eye_angle) < 10:
                                score += 10 - abs(eye_angle)
                except Exception as e:
                    # If mesh fails, continue with current score
                    pass
                
                # Update best angle if this one scores better
                if score > max_score:
                    max_score = score
                    best_angle = angle
        
        # If a face was found, we may need to fine-tune the rotation
        if face_found:
            # If we found a face but it's one of the non-90 degree angles, 
            # or if it's 0 but the face isn't quite straight
            if best_angle in [45, 135, 225, 315] or (best_angle == 0 and max_score < 15):
                # Apply initial rotation if needed
                if best_angle > 0:
                    rotated_image = self.rotate_image_by_angle(image, best_angle) if best_angle in [90, 180, 270] else self.rotate_image(image, best_angle)
                else:
                    rotated_image = image.copy()
                
                # Now fine-tune the angle using face landmarks
                fine_angle = self.get_precise_face_angle(rotated_image)
                
                # Combine the coarse and fine angles
                return best_angle - fine_angle
            else:
                return best_angle
        
        # If no face found, return 0 (no rotation)
        return 0
    
    def get_precise_face_angle(self, image):
        """Get a precise face angle using facial landmarks"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            # Try face detection as fallback
            detection_results = self.face_detection.process(rgb_image)
            if detection_results.detections:
                # Use the nose and eyes from face detection
                detection = detection_results.detections[0]
                landmarks = detection.location_data.relative_keypoints
                
                # Get eye landmarks
                left_eye = landmarks[0]
                right_eye = landmarks[1]
                
                # Calculate angle
                dx = right_eye.x - left_eye.x
                dy = right_eye.y - left_eye.y
                
                # Return angle in degrees
                return np.degrees(np.arctan2(dy, dx))
            
            return 0
        
        # Use face mesh landmarks for more precise angle
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Get key points for angle measurement
        left_eye = face_landmarks[33]  # left eye
        right_eye = face_landmarks[263]  # right eye
        
        # Calculate angle
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        angle = np.degrees(np.arctan2(dy, dx))
        
        return angle
    
    def detect_face_orientation(self, image):
        """
        Legacy method - now just calls get_face_orientation_angle
        """
        return self.get_face_orientation_angle(image)
    
    def rotate_image_by_angle(self, image, angle):
        """Rotate image by a specific angle (90, 180, 270 degrees)"""
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
    
    def remove_black_borders(self, image):
        """
        Remove black borders after rotation by cropping to the non-black region
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find non-black regions
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add a small margin
        margin = 2
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2*margin)
        h = min(image.shape[0] - y, h + 2*margin)
        
        # Crop image to this rectangle
        cropped = image[y:y+h, x:x+w]
        
        return cropped
    
    def get_face_angle(self, image):
        """Legacy method - now just calls get_precise_face_angle"""
        return self.get_precise_face_angle(image)

    def rotate_image(self, image, angle):
        """Rotate image by a precise angle with minimal quality loss"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions
        cos = np.abs(rot_matrix[0, 0])
        sin = np.abs(rot_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust translation to ensure the whole image is visible
        rot_matrix[0, 2] += (new_w / 2) - center[0]
        rot_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Perform the rotation with the adjusted dimensions
        rotated = cv2.warpAffine(image, rot_matrix, (new_w, new_h), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0, 0, 0))
        
        return rotated
    
    def process_photos(self):
        if self.processing:
            return
        
        # Get settings
        input_dir = self.input_folder.get()
        output_dir = self.output_folder.get()
        target_width = self.width.get()
        target_height = self.height.get()
        
        # Validate
        if not os.path.isdir(input_dir):
            messagebox.showerror("Error", f"Input directory does not exist: {input_dir}")
            return
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of images to process
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        if not image_files:
            messagebox.showinfo("Info", f"No images found in {input_dir}. Please add some images.")
            return
        
        # Start processing thread
        self.processing = True
        self.process_button.state(['disabled'])
        self.update_status("Processing started...")
        self.progress_bar['maximum'] = len(image_files)
        self.progress_bar['value'] = 0
        
        # Run processing in a separate thread
        threading.Thread(target=self.process_images_thread, 
                        args=(input_dir, output_dir, (target_width, target_height), image_files)).start()
    
    def process_images_thread(self, input_dir, output_dir, target_size, image_files):
        processed_count = 0
        success_count = 0
        
        for filename in image_files:
            try:
                self.root.after(0, lambda file=filename: self.update_status(f"Processing {file}..."))
                
                # Read image
                path = os.path.join(input_dir, filename)
                image = cv2.imread(path)
                if image is None:
                    self.log(f"Could not read file: {filename}")
                    continue
                
                # Detect and correct image orientation (handles arbitrary angles now)
                orientation_angle = self.get_face_orientation_angle(image)
                
                if orientation_angle != 0:
                    self.log(f"Correcting image orientation by {orientation_angle:.1f} degrees")
                    
                    # Choose rotation method based on angle
                    if orientation_angle in [90, 180, 270]:
                        corrected_image = self.rotate_image_by_angle(image, orientation_angle)
                    else:
                        corrected_image = self.rotate_image(image, orientation_angle)
                        
                    # Remove any black borders
                    corrected_image = self.remove_black_borders(corrected_image)
                else:
                    corrected_image = image
                
                # For any special cases where orientation wasn't fully fixed
                # Try to detect if there's still a slight angle that needs correction
                fine_angle = self.get_precise_face_angle(corrected_image)
                
                if abs(fine_angle) > 2:  # Only correct if angle is significant
                    self.log(f"Fine-tuning rotation by {-fine_angle:.1f} degrees")
                    rotated_image = self.rotate_image(corrected_image, -fine_angle)
                    rotated_image = self.remove_black_borders(rotated_image)
                else:
                    rotated_image = corrected_image
                
                # Detect person using YOLO
                results = self.model(rotated_image)
                boxes = results[0].boxes.data.cpu().numpy()
                person_boxes = [box for box in boxes if int(box[5]) == 0]  # Class 0 is person
                
                if not person_boxes:
                    self.log(f"No person detected in: {filename}, using the entire image")
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
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, resized)
                self.log(f"Processed {filename} ✅")
                success_count += 1
                
            except Exception as e:
                self.log(f"Error processing {filename}: {str(e)}")
                # Fallback: just resize the original image if processing fails
                try:
                    # Try a simple rotation if it failed with complex methods
                    detection_results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if detection_results.detections:
                        # Use simple orientation correction based on face detection
                        h, w = image.shape[:2]
                        detection = detection_results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        
                        # Check if image is likely rotated 90 degrees
                        face_h = bbox.height * h
                        face_w = bbox.width * w
                        
                        if face_w > face_h * 1.2 and (bbox.ymin < 0.3 or bbox.ymin + bbox.height > 0.7):
                            # Face is wider than tall and not centered vertically - likely 90° rotated
                            image = self.rotate_image_by_angle(image, 90)
                    
                    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                    save_path = os.path.join(output_dir, filename)
                    cv2.imwrite(save_path, resized)
                    self.log(f"Fallback: Resized {filename} without processing ✅")
                    success_count += 1
                except Exception as fallback_error:
                    self.log(f"Fallback also failed: {str(fallback_error)}")
            
            processed_count += 1
            self.root.after(0, lambda count=processed_count: self.progress_bar.config(value=count))
        
        # Processing completed
        self.root.after(0, lambda: self.update_status(f"Processing completed. {success_count} of {processed_count} images processed successfully."))
        self.root.after(0, lambda: self.process_button.state(['!disabled']))
        self.processing = False

if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoEditorApp(root)
    root.mainloop() 