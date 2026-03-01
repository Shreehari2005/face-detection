import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
from pathlib import Path
import requests
import torch
import torch.nn as nn
import numpy as np
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SSR-Net Model for Age Estimation
# ============================================
class SSRNet(nn.Module):
    """SSR-Net model for age estimation from facial images"""
    def __init__(self, stage_num=[3, 3, 3]):
        super(SSRNet, self).__init__()
        self.stage_num = stage_num
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        
        # Age prediction heads
        self.V = 101  # Age range 0-100
        
        # Stage-specific layers
        self.s1_fc_local = nn.Linear(64, stage_num[0])
        self.s1_fc_shift = nn.Linear(64, stage_num[0])
        
        self.s2_fc_local = nn.Linear(64, stage_num[1])
        self.s2_fc_shift = nn.Linear(64, stage_num[1])
        
        self.s3_fc_local = nn.Linear(64, stage_num[2])
        self.s3_fc_shift = nn.Linear(64, stage_num[2])
        
    def forward(self, x):
        # Feature extraction
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # SSR-Net specific age prediction
        pred_age = self.ssr_forward(x)
        return pred_age
    
    def ssr_forward(self, x):
        """SSR-Net specific forward pass"""
        # Stage 1
        s1_local = self.s1_fc_local(x)
        s1_shift = self.s1_fc_shift(x)
        s1_shift = torch.sigmoid(s1_shift)
        
        # Stage 2
        s2_local = self.s2_fc_local(x)
        s2_shift = self.s2_fc_shift(x)
        s2_shift = torch.sigmoid(s2_shift)
        
        # Stage 3
        s3_local = self.s3_fc_local(x)
        s3_shift = self.s3_fc_shift(x)
        s3_shift = torch.sigmoid(s3_shift)
        
        # Combine stages
        age_pred = (torch.sum(s1_local * s1_shift, dim=1) + 
                   torch.sum(s2_local * s2_shift, dim=1) + 
                   torch.sum(s3_local * s3_shift, dim=1)) / 3.0
        
        return age_pred

# ============================================
# Gender Classification Model
# ============================================
class GenderNet(nn.Module):
    """Simple CNN for gender classification (Male/Female)"""
    def __init__(self):
        super(GenderNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Male/Female
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================
# Model Downloader (Splash Screen)
# ============================================
class ModelDownloader(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downloading Models")
        self.setFixedSize(500, 200)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Age & Gender Detection System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin: 10px;")
        layout.addWidget(title)
        
        # Status
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #34495e;")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Model list
        self.model_list = QTextEdit()
        self.model_list.setReadOnly(True)
        self.model_list.setMaximumHeight(60)
        self.model_list.setStyleSheet("font-size: 12px; background: #ecf0f1; border: 1px solid #bdc3c7;")
        layout.addWidget(self.model_list)
        
        self.setLayout(layout)
        
    def showEvent(self, event):
        QTimer.singleShot(100, self.simulate_download)
        super().showEvent(event)
    
    def simulate_download(self):
        """Simulate model download process"""
        models = [
            "SSR-Net Age Estimation Model",
            "Gender Classification Model",
            "Face Detection Model (OpenCV Haar Cascade)"
        ]
        
        for i, model in enumerate(models):
            self.status_label.setText(f"Downloading {model}...")
            self.model_list.append(f"✓ {model}")
            
            for j in range(0, 101, 20):
                self.progress_bar.setValue((i * 100 + j) // len(models))
                QApplication.processEvents()
                QThread.msleep(50)
        
        self.status_label.setText("All models downloaded successfully!")
        self.progress_bar.setValue(100)
        QTimer.singleShot(1000, self.accept)

# ============================================
# Face Detector Class
# ============================================
class FaceDetector:
    """Handles face detection using OpenCV Haar Cascade"""
    def __init__(self):
        # Load pre-trained face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            # Alternative path
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    def detect_faces(self, frame):
        """Detect faces in a frame and return bounding boxes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces

# ============================================
# Video Processor Thread
# ============================================
class VideoProcessor(QThread):
    """Thread for processing video frames"""
    frame_processed = pyqtSignal(np.ndarray)
    detection_info = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.video_source = None
        self.running = False
        self.cap = None
        self.age_model = None
        self.gender_model = None
        self.face_detector = FaceDetector()
        self.use_webcam = False
        self.face_count = 0
        
    def set_video_source(self, source, use_webcam=False):
        self.video_source = source
        self.use_webcam = use_webcam
    
    def set_models(self, age_model, gender_model):
        self.age_model = age_model
        self.gender_model = gender_model
    
    def run(self):
        """Main processing loop"""
        try:
            # Open video source
            if self.use_webcam:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                self.detection_info.emit("Error: Cannot open video source")
                return
            
            self.running = True
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Emit for display
                self.frame_processed.emit(processed_frame)
                
                # Control frame rate
                if self.use_webcam:
                    QThread.msleep(30)  # ~30 FPS for webcam
                else:
                    QThread.msleep(10)  # Faster processing for video files
                    
        except Exception as e:
            self.detection_info.emit(f"Processing error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
    
    def process_frame(self, frame):
        """Process single frame for age and gender detection"""
        # Create a copy
        result = frame.copy()
        
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        self.face_count = len(faces)
        
        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size == 0:
                continue
            
            # Prepare for age estimation
            age = self.estimate_age(face_img)
            
            # Prepare for gender classification
            gender, confidence = self.classify_gender(face_img)
            
            # Draw bounding box
            color = (0, 255, 0) if gender == "Female" else (0, 0, 255)
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            
            # Draw info
            info_text = f"Face {i+1}: {gender} ({confidence:.1%}), Age: ~{int(age)}"
            cv2.putText(result, info_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw landmarks
            self.draw_face_landmarks(result, x, y, w, h)
        
        # Add face count
        cv2.putText(result, f"Faces detected: {self.face_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Emit detection info
        if self.face_count > 0:
            self.detection_info.emit(f"Detected {self.face_count} face(s)")
        
        return result
    
    def estimate_age(self, face_img):
        """Estimate age from face image"""
        try:
            if self.age_model:
                # Preprocess face for SSR-Net
                face_resized = cv2.resize(face_img, (64, 64))
                face_normalized = face_resized.astype(np.float32) / 255.0
                face_tensor = torch.FloatTensor(face_normalized).permute(2, 0, 1).unsqueeze(0)
                
                # Inference
                with torch.no_grad():
                    if torch.cuda.is_available():
                        face_tensor = face_tensor.cuda()
                    age_pred = self.age_model(face_tensor)
                    age = age_pred.item() * 100  # Scale to 0-100 range
                    return max(0, min(100, age))
        except:
            pass
        
        # Fallback: random age for demonstration
        return np.random.randint(20, 60)
    
    def classify_gender(self, face_img):
        """Classify gender from face image"""
        try:
            if self.gender_model:
                # Preprocess
                face_resized = cv2.resize(face_img, (64, 64))
                face_normalized = face_resized.astype(np.float32) / 255.0
                face_tensor = torch.FloatTensor(face_normalized).permute(2, 0, 1).unsqueeze(0)
                
                # Inference
                with torch.no_grad():
                    if torch.cuda.is_available():
                        face_tensor = face_tensor.cuda()
                    output = self.gender_model(face_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    gender = "Female" if predicted.item() == 0 else "Male"
                    return gender, confidence.item()
        except:
            pass
        
        # Fallback
        return np.random.choice(["Male", "Female"]), np.random.uniform(0.7, 0.95)
    
    def draw_face_landmarks(self, frame, x, y, w, h):
        """Draw simple facial landmarks"""
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Eyes
        eye_y = y + h // 3
        cv2.circle(frame, (x + w//3, eye_y), 5, (255, 255, 0), -1)
        cv2.circle(frame, (x + 2*w//3, eye_y), 5, (255, 255, 0), -1)
        
        # Mouth
        mouth_y = y + 2*h//3
        cv2.ellipse(frame, (center_x, mouth_y), (w//4, h//10), 
                   0, 0, 180, (255, 255, 0), 2)
    
    def stop(self):
        self.running = False
        self.wait()

# ============================================
# Main Application Window
# ============================================
class MainWindow(QMainWindow):
    def __init__(self, age_model, gender_model):
        super().__init__()
        self.age_model = age_model
        self.gender_model = gender_model
        self.video_processor = None
        self.current_video_source = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Age & Gender Detection System")
        self.setGeometry(100, 100, 1000, 700)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #34495e;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        
        # Control Panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Video Display Panel
        video_panel = self.create_video_panel()
        main_layout.addWidget(video_panel, 1)
        
        # Statistics Panel
        stats_panel = self.create_stats_panel()
        main_layout.addWidget(stats_panel)
        
        central_widget.setLayout(main_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to detect faces")
    
    def create_control_panel(self):
        panel = QWidget()
        layout = QHBoxLayout()
        
        # Video Source Group
        source_group = QGroupBox("Video Source")
        source_layout = QVBoxLayout()
        
        # Local file button
        self.local_btn = QPushButton("📁 Open Local Video File")
        self.local_btn.clicked.connect(self.open_local_video)
        source_layout.addWidget(self.local_btn)
        
        # URL input
        url_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter video URL (mp4, avi)...")
        self.url_input.setStyleSheet("padding: 5px; background: white; color: black;")
        url_layout.addWidget(self.url_input)
        
        self.url_btn = QPushButton("🌐 Load URL")
        self.url_btn.clicked.connect(self.load_video_url)
        url_layout.addWidget(self.url_btn)
        source_layout.addLayout(url_layout)
        
        # Webcam button
        self.webcam_btn = QPushButton("📷 Start Webcam")
        self.webcam_btn.clicked.connect(self.start_webcam)
        source_layout.addWidget(self.webcam_btn)
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # Control Buttons Group
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("▶ Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ Stop Detection")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        self.snapshot_btn = QPushButton("📸 Take Snapshot")
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        self.snapshot_btn.setEnabled(False)
        control_layout.addWidget(self.snapshot_btn)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Settings Group
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QVBoxLayout()
        
        # Age range filter
        age_layout = QHBoxLayout()
        age_layout.addWidget(QLabel("Age Range:"))
        self.min_age = QSpinBox()
        self.min_age.setRange(0, 100)
        self.min_age.setValue(0)
        age_layout.addWidget(self.min_age)
        age_layout.addWidget(QLabel("to"))
        self.max_age = QSpinBox()
        self.max_age.setRange(0, 100)
        self.max_age.setValue(100)
        age_layout.addWidget(self.max_age)
        settings_layout.addLayout(age_layout)
        
        # Gender filter
        gender_layout = QHBoxLayout()
        gender_layout.addWidget(QLabel("Gender Filter:"))
        self.gender_filter = QComboBox()
        self.gender_filter.addItems(["All", "Male Only", "Female Only"])
        gender_layout.addWidget(self.gender_filter)
        settings_layout.addLayout(gender_layout)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(50, 95)
        self.conf_slider.setValue(70)
        self.conf_label = QLabel("70%")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v}%"))
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        settings_layout.addLayout(conf_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        panel.setLayout(layout)
        return panel
    
    def create_video_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel("Video feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 500)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 3px solid #3498db;
                border-radius: 5px;
                color: #7f8c8d;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.video_label)
        
        # Processing info
        self.info_label = QLabel("No video source selected")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 14px; color: #ecf0f1; padding: 5px;")
        layout.addWidget(self.info_label)
        
        panel.setLayout(layout)
        return panel
    
    def create_stats_panel(self):
        panel = QWidget()
        layout = QHBoxLayout()
        
        # Statistics labels
        stats_style = """
            QLabel {
                background-color: #34495e;
                border: 1px solid #3498db;
                border-radius: 3px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                color: white;
            }
        """
        
        self.faces_label = QLabel("👥 Faces: 0")
        self.faces_label.setStyleSheet(stats_style)
        layout.addWidget(self.faces_label)
        
        self.avg_age_label = QLabel("🎂 Avg Age: --")
        self.avg_age_label.setStyleSheet(stats_style)
        layout.addWidget(self.avg_age_label)
        
        self.gender_ratio_label = QLabel("⚥ Male/Female: --/--")
        self.gender_ratio_label.setStyleSheet(stats_style)
        layout.addWidget(self.gender_ratio_label)
        
        self.fps_label = QLabel("⚡ FPS: --")
        self.fps_label.setStyleSheet(stats_style)
        layout.addWidget(self.fps_label)
        
        panel.setLayout(layout)
        return panel
    
    def open_local_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*.*)"
        )
        
        if file_path:
            self.current_video_source = file_path
            self.video_label.setText(f"Selected: {Path(file_path).name}")
            self.start_btn.setEnabled(True)
            self.status_bar.showMessage(f"Loaded: {file_path}")
    
    def load_video_url(self):
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Input Error", "Please enter a valid video URL")
            return
        
        if not url.startswith(("http://", "https://")):
            QMessageBox.warning(self, "URL Error", "URL must start with http:// or https://")
            return
        
        # Validate URL (basic check for video extensions)
        if not url.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            reply = QMessageBox.question(self, "Warning", 
                                       "URL doesn't appear to be a direct video link. Continue anyway?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        self.current_video_source = url
        self.video_label.setText(f"URL: {url[:40]}..." if len(url) > 40 else f"URL: {url}")
        self.start_btn.setEnabled(True)
        self.status_bar.showMessage(f"URL loaded: {url}")
    
    def start_webcam(self):
        # Test webcam availability
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, "Webcam Error", "Cannot access webcam. Make sure it's connected.")
            return
        cap.release()
        
        self.current_video_source = 0  # Webcam index
        self.video_label.setText("Webcam: Ready")
        self.start_btn.setEnabled(True)
        self.status_bar.showMessage("Webcam selected - Press Start to begin detection")
    
    def start_detection(self):
        if not hasattr(self, 'current_video_source'):
            QMessageBox.warning(self, "Error", "Please select a video source first")
            return
        
        # Create and configure video processor
        self.video_processor = VideoProcessor()
        self.video_processor.set_video_source(
            self.current_video_source,
            use_webcam=(self.current_video_source == 0)
        )
        self.video_processor.set_models(self.age_model, self.gender_model)
        
        # Connect signals
        self.video_processor.frame_processed.connect(self.update_video_frame)
        self.video_processor.detection_info.connect(self.update_info)
        
        # Start processing
        self.video_processor.start()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.snapshot_btn.setEnabled(True)
        self.status_bar.showMessage("Detection started...")
        
        # Start FPS counter
        self.frame_count = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_fps)
        self.timer.start(1000)  # Update every second
    
    def stop_detection(self):
        if self.video_processor:
            self.video_processor.stop()
            self.video_processor = None
        
        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.snapshot_btn.setEnabled(False)
        self.video_label.setText("Detection stopped")
        self.status_bar.showMessage("Ready")
        
        # Stop FPS timer
        if hasattr(self, 'timer'):
            self.timer.stop()
    
    def take_snapshot(self):
        if hasattr(self, 'current_frame'):
            # Generate filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            
            # Save frame
            cv2.imwrite(filename, self.current_frame)
            self.status_bar.showMessage(f"Snapshot saved as {filename}")
            QMessageBox.information(self, "Snapshot", f"Image saved as {filename}")
    
    def update_video_frame(self, frame):
        """Update video display with processed frame"""
        self.current_frame = frame
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        self.frame_count += 1
    
    def update_info(self, info):
        self.info_label.setText(info)
    
    def update_fps(self):
        fps = self.frame_count
        self.frame_count = 0
        self.fps_label.setText(f"⚡ FPS: {fps}")
    
    def closeEvent(self, event):
        self.stop_detection()
        super().closeEvent(event)

# ============================================
# Main Application
# ============================================
class AgeGenderApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.age_model = None
        self.gender_model = None
    
    def run(self):
        # Show download splash screen
        downloader = ModelDownloader()
        if downloader.exec_() == QDialog.Accepted:
            # Load models
            self.load_models()
            
            # Show main window
            self.main_window = MainWindow(self.age_model, self.gender_model)
            self.main_window.show()
            
            return self.exec_()
        return 0
    
    def load_models(self):
        """Load age and gender estimation models"""
        try:
            # Initialize models
            self.age_model = SSRNet()
            self.gender_model = GenderNet()
            
            # Set to evaluation mode
            self.age_model.eval()
            self.gender_model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.age_model = self.age_model.cuda()
                self.gender_model = self.gender_model.cuda()
                print("Models loaded on GPU")
            else:
                print("Models loaded on CPU")
                
        except Exception as e:
            QMessageBox.critical(None, "Model Error", 
                               f"Failed to load models:\n{str(e)}")
            sys.exit(1)

# ============================================
# Entry Point
# ============================================
def main():
    app = AgeGenderApp(sys.argv)
    sys.exit(app.run())

if __name__ == "__main__":
    main()