import sys
import cv2
import os
import time
from datetime import datetime
import numpy as np
import torch
import mediapipe as mp
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QProgressBar, QGroupBox,
                           QApplication, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer, QDateTime
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
class HumanMeshNet(nn.Module):
    """Neural network for 3D face mesh prediction"""
    def __init__(self, num_vertices=1000):
        super(HumanMeshNet, self).__init__()

        # Encoder - takes image features and processes them
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        # Calculate flattened size for dense layers
        self.flatten_size = 512 * 8 * 8

        # Mesh vertex predictor
        self.mesh_predictor = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_vertices * 3)  # 3 coordinates per vertex
        )

    def forward(self, x):
        """Forward pass through the network"""
        try:
            batch_size = x.shape[0]

            # Extract features
            x = self.encoder(x)
            x = self.fusion(x)

            # Flatten and predict vertices
            x = x.view(batch_size, -1)
            vertices = self.mesh_predictor(x)

            # Reshape to (batch_size, num_vertices, 3)
            return vertices.view(batch_size, -1, 3)

        except Exception as e:
            print(f"Error in HumanMeshNet forward pass: {e}")
            return None

    def preprocess_image(self, image):
        """Preprocess image for network input"""
        try:
            if isinstance(image, np.ndarray):
                # Convert numpy array to tensor
                image = torch.from_numpy(image).float()

            # Ensure correct dimensions
            if len(image.shape) == 3:
                image = image.unsqueeze(0)

            # Normalize if needed
            if image.max() > 1.0:
                image = image / 255.0

            return image

        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
class LaplacianSmoother:
    def __init__(self):
        self.adjacency_matrix = None

    def compute_adjacency_matrix(self, faces, num_vertices):
        try:
            # Ensure consistent dtype
            adjacency = torch.zeros((num_vertices, num_vertices),
                                 dtype=torch.float32,
                                 device=faces.device)
            faces = faces.long()

            for face in faces:
                if len(face) >= 3:
                    v0, v1, v2 = face[0], face[1], face[2]
                    if max(v0, v1, v2) < num_vertices:
                        adjacency[v0, v1] = adjacency[v1, v0] = 1
                        adjacency[v1, v2] = adjacency[v2, v1] = 1
                        adjacency[v2, v0] = adjacency[v0, v2] = 1

            return adjacency
        except Exception as e:
            print(f"Error computing adjacency matrix: {e}")
            return torch.eye(num_vertices, dtype=torch.float32, device=faces.device)

    def smooth(self, vertices, faces, iterations=3, lambda_factor=0.5):
        try:
            # Ensure vertices are float32
            vertices = vertices.to(dtype=torch.float32)

            if self.adjacency_matrix is None or self.adjacency_matrix.shape[0] != vertices.shape[0]:
                self.adjacency_matrix = self.compute_adjacency_matrix(
                    faces, vertices.shape[0]).to(dtype=torch.float32, device=vertices.device)

            smoothed_vertices = vertices.clone()
            for _ in range(iterations):
                degrees = self.adjacency_matrix.sum(dim=1, keepdim=True).clamp(min=1)
                laplacian = torch.matmul(self.adjacency_matrix, smoothed_vertices)
                laplacian = laplacian / degrees
                smoothed_vertices = smoothed_vertices + lambda_factor * (laplacian - smoothed_vertices)

            return smoothed_vertices
        except Exception as e:
            print(f"Error in smoothing: {e}")
            return vertices
class GLMeshWidget(QGLWidget):
    """Enhanced OpenGL widget with improved error handling and performance"""
    def __init__(self, parent=None):
        super(GLMeshWidget, self).__init__(parent)
        self.mesh_vertices = None
        self.mesh_faces = None
        self.mesh_colors = None
        self.rotation = [0.0, 0.0, 0.0]
        self.translation = [0.0, 0.0, -5.0]
        self.scale = 1.0
        self.setMinimumSize(400, 400)
        self.initializeGL()
        self.start_rotation_timer()

    def start_rotation_timer(self):
        self.rotation_timer = QTimer(self)
        self.rotation_timer.timeout.connect(self.rotate)
        self.rotation_timer.start(50)

    def rotate(self):
        self.rotation[1] += 2.0
        self.updateGL()

    def initializeGL(self):
        try:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

            # Enhanced lighting setup
            glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
            glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])

            # Material properties
            glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
            glMaterialf(GL_FRONT, GL_SHININESS, 50.0)

            glClearColor(0.9, 0.9, 0.9, 1.0)
        except Exception as e:
            print(f"Error initializing OpenGL: {e}")

    def paintGL(self):
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            glTranslatef(*self.translation)
            glRotatef(self.rotation[0], 1.0, 0.0, 0.0)
            glRotatef(self.rotation[1], 0.0, 1.0, 0.0)
            glRotatef(self.rotation[2], 0.0, 0.0, 1.0)
            glScalef(self.scale, self.scale, self.scale)

            if self.mesh_vertices is not None and self.mesh_faces is not None:
                # Draw filled mesh
                glEnable(GL_LIGHTING)
                glBegin(GL_TRIANGLES)
                for face in self.mesh_faces:
                    for idx in face:
                        if idx < len(self.mesh_vertices):
                            if self.mesh_colors is not None and idx < len(self.mesh_colors):
                                color = self.mesh_colors[idx]
                                glColor3f(color[0]/255.0, color[1]/255.0, color[2]/255.0)
                            else:
                                glColor3f(0.7, 0.7, 0.7)
                            vertex = self.mesh_vertices[idx]
                            glVertex3f(*vertex)
                glEnd()

                # Draw wireframe
                glDisable(GL_LIGHTING)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glColor3f(0.2, 0.2, 0.2)
                glBegin(GL_TRIANGLES)
                for face in self.mesh_faces:
                    for idx in face:
                        if idx < len(self.mesh_vertices):
                            vertex = self.mesh_vertices[idx]
                            glVertex3f(*vertex)
                glEnd()
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glEnable(GL_LIGHTING)

        except Exception as e:
            print(f"Error in paintGL: {e}")

    def resizeGL(self, w, h):
        try:
            glViewport(0, 0, w, h)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45.0, float(w)/float(h), 0.1, 100.0)
            glMatrixMode(GL_MODELVIEW)
        except Exception as e:
            print(f"Error in resizeGL: {e}")

    def update_mesh(self, vertices, faces, colors=None):
        try:
            if vertices is not None and len(vertices) > 0:
                vertices = np.array(vertices, dtype=np.float32)
                center = np.mean(vertices, axis=0)
                vertices = vertices - center
                scale = np.max(np.abs(vertices))
                if scale > 0:
                    vertices = vertices / scale
                self.mesh_vertices = vertices
                self.mesh_faces = faces
                self.mesh_colors = colors
                self.updateGL()
        except Exception as e:
            print(f"Error updating mesh: {e}")

class FaceScanner(QMainWindow):
    """Enhanced main application window with improved error handling and UI"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Face Scanner")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize state variables
        self.is_recording = False
        self.frames = []
        self.estimated_meshes = []
        self.current_mesh = None
        self.final_mesh = None
        self.final_colors = None

        # Initialize deep learning components first
        self.setup_deep_learning()

        # Then initialize template mesh
        self.template_vertices, self.template_faces = self.create_face_mesh()

        # Set up other components
        self.setup_camera_calibration()
        self.initialize_face_mesh()
        self.setup_ui()  # UI setup should come after other initializations
        self.initialize_camera()

        # Show initial mesh if available
        if self.template_vertices is not None and self.template_faces is not None:
            self.gl_widget.update_mesh(
                self.template_vertices.cpu().numpy(),
                self.template_faces.cpu().numpy(),
                np.ones((len(self.template_vertices), 3)) * 180
            )

    def setup_deep_learning(self):
        """Initialize deep learning components with error handling"""
        try:
            # Set up CUDA if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

            # Initialize model
            self.mesh_model = HumanMeshNet().to(self.device)

            # Initialize smoother
            self.smoother = LaplacianSmoother()

            # Set up preprocessing transform
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        except Exception as e:
            print(f"Error setting up deep learning components: {e}")
            QMessageBox.warning(self, "Warning",
                              "Could not initialize deep learning components. Some features may be limited.")
    def initialize_face_mesh(self):
        """Initialize MediaPipe face mesh with proper configuration"""
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_landmarks=True
            )
        except Exception as e:
            print(f"Error initializing face mesh: {e}")
            QMessageBox.critical(self, "Error",
                               "Could not initialize face tracking. Please check your MediaPipe installation.")

    def setup_camera_calibration(self):
        """Set up camera calibration parameters"""
        try:
            # Enhanced camera matrix with more realistic parameters
            focal_length = 1000.0
            center_x = 640.0
            center_y = 480.0

            self.camera_matrix = np.array([
                [focal_length, 0, center_x],
                [0, focal_length, center_y],
                [0, 0, 1]
            ], dtype=np.float32)

            # Add distortion coefficients
            self.dist_coeffs = np.zeros(5, dtype=np.float32)

            # Camera pose
            self.camera_rotation = np.eye(3)
            self.camera_translation = np.array([0, 0, -2])

        except Exception as e:
            print(f"Error in camera calibration setup: {e}")
            self.statusBar().showMessage(f"Error generating 3D model: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to generate 3D model: {str(e)}")

    def normalize_vertices(self, vertices):
        """Normalize vertex positions with improved scaling"""
        try:
            # Center the mesh
            center = np.mean(vertices, axis=0)
            vertices = vertices - center

            # Scale to unit sphere while preserving aspect ratio
            scale = np.max(np.abs(vertices))
            if scale > 0:
                vertices = vertices / scale

            # Apply face-specific constraints
            face_height_ratio = 1.5  # Typical face height/width ratio
            vertices[:, 1] *= face_height_ratio

            return vertices

        except Exception as e:
            print(f"Error normalizing vertices: {e}")
            return vertices

    def save_model(self):
        """Save the generated 3D model with enhanced file format support"""
        try:
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Save 3D Model",
                f"face_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "PLY Files (*.ply);;OBJ Files (*.obj)"
            )

            if not file_path:
                return

            self.statusBar().showMessage("Saving 3D model...")
            self.progress_bar.setValue(0)

            # Determine file format
            if file_path.endswith('.ply'):
                self.save_ply(file_path)
            elif file_path.endswith('.obj'):
                self.save_obj(file_path)
            else:
                file_path += '.ply'  # Default to PLY
                self.save_ply(file_path)

            self.statusBar().showMessage("3D model saved successfully!")
            self.progress_bar.setValue(100)

        except Exception as e:
            print(f"Error saving model: {e}")
            self.statusBar().showMessage(f"Error saving model: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")

    def save_ply(self, file_path):
        """Save model in PLY format with enhanced attributes"""
        try:
            with open(file_path, 'w') as f:
                # Write header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(self.final_mesh)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("property float confidence\n")  # Add confidence measure
                f.write(f"element face {len(self.template_faces)}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")

                # Write vertices with colors and confidence
                for i, (vertex, color) in enumerate(zip(self.final_mesh, self.final_colors)):
                    # Calculate confidence based on position
                    confidence = 1.0 - np.abs(vertex[2])  # Higher confidence for front-facing vertices
                    f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} "
                           f"{int(color[0])} {int(color[1])} {int(color[2])} "
                           f"{confidence:.6f}\n")

                # Write faces
                faces = self.template_faces.cpu().numpy()
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        except Exception as e:
            raise Exception(f"Failed to save PLY file: {str(e)}")

    def save_obj(self, file_path):
        """Save model in OBJ format with material support"""
        try:
            # Create material file
            mtl_path = file_path.replace('.obj', '.mtl')
            with open(mtl_path, 'w') as f:
                f.write("newmtl FaceMaterial\n")
                f.write("Ka 1.000 1.000 1.000\n")  # Ambient color
                f.write("Kd 0.800 0.800 0.800\n")  # Diffuse color
                f.write("Ks 0.500 0.500 0.500\n")  # Specular color
                f.write("Ns 50.0\n")  # Specular exponent
                f.write("d 1.0\n")  # Opacity

            # Save OBJ file
            with open(file_path, 'w') as f:
                f.write(f"mtllib {os.path.basename(mtl_path)}\n")
                f.write("o FaceMesh\n")

                # Write vertices
                for vertex in self.final_mesh:
                    f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

                # Write texture coordinates (if available)
                for _ in range(len(self.final_mesh)):
                    f.write("vt 0.0 0.0\n")

                # Write normals
                self.compute_and_write_normals(f)

                # Write faces with material
                f.write("usemtl FaceMaterial\n")
                faces = self.template_faces.cpu().numpy()
                for face in faces:
                    # OBJ indices are 1-based
                    f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                           f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                           f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")

        except Exception as e:
            raise Exception(f"Failed to save OBJ file: {str(e)}")

    def compute_and_write_normals(self, file):
        """Compute and write vertex normals"""
        try:
            vertices = self.final_mesh
            faces = self.template_faces.cpu().numpy()

            # Initialize normal array
            normals = np.zeros_like(vertices)

            # Compute face normals and accumulate to vertices
            for face in faces:
                v0, v1, v2 = vertices[face]
                normal = np.cross(v1 - v0, v2 - v0)
                normal = normal / np.linalg.norm(normal)

                # Add to all vertices of this face
                normals[face] += normal

            # Normalize accumulated normals
            norms = np.linalg.norm(normals, axis=1)
            mask = norms > 0
            normals[mask] = normals[mask] / norms[mask, np.newaxis]

            # Write to file
            for normal in normals:
                file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

        except Exception as e:
            print(f"Error computing normals: {e}")
            # Write default normals if computation fails
            for _ in range(len(self.final_mesh)):
                file.write("vn 0.0 1.0 0.0\n")

    def display_frame(self, frame):
        """Display frame in the UI with proper scaling"""
        try:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"Error displaying frame: {e}")

    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        try:
            # Stop timers
            if hasattr(self, 'timer'):
                self.timer.stop()
            if hasattr(self, 'gl_widget') and hasattr(self.gl_widget, 'rotation_timer'):
                self.gl_widget.rotation_timer.stop()

            # Release camera
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()

            # Clean up MediaPipe resources
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()

            # Clean up OpenGL context
            if hasattr(self, 'gl_widget'):
                self.gl_widget.makeCurrent()
                self.gl_widget.doneCurrent()

            event.accept()

        except Exception as e:
            print(f"Error during cleanup: {e}")
            event.accept()

    def initialize_camera(self):
        """Initialize camera with error handling and automatic device selection"""
        try:
            # Try to find the best available camera
            for i in range(3):  # Try first 3 camera indices
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    # Try to set high resolution
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)

                    # Verify if camera is working by reading a frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        break
                    else:
                        self.cap.release()

            if not self.cap.isOpened():
                raise Exception("No working camera found")

            # Start frame update timer
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(33)  # ~30 FPS

            return True

        except Exception as e:
            print(f"Camera initialization error: {e}")
            QMessageBox.critical(self, "Error", f"Could not initialize camera: {str(e)}")
            return False

    def setup_ui(self):
        """Set up the user interface with improved layout and controls"""
        try:
            # Create main widget and layout
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            layout = QVBoxLayout(main_widget)

            # Create preview layout
            preview_layout = QHBoxLayout()

            # Camera preview group
            camera_group = QGroupBox("Camera Feed")
            camera_layout = QVBoxLayout(camera_group)
            self.image_label = QLabel()
            self.image_label.setMinimumSize(640, 480)
            self.image_label.setAlignment(Qt.AlignCenter)
            camera_layout.addWidget(self.image_label)
            preview_layout.addWidget(camera_group)

            # 3D preview group
            model_group = QGroupBox("3D Model Preview")
            model_layout = QVBoxLayout(model_group)
            self.gl_widget = GLMeshWidget()
            model_layout.addWidget(self.gl_widget)
            preview_layout.addWidget(model_group)

            layout.addLayout(preview_layout)

            # Controls layout
            controls_layout = QHBoxLayout()

            # Scan button
            self.scan_button = QPushButton("Start Scan")
            self.scan_button.clicked.connect(self.toggle_scanning)
            self.scan_button.setMinimumWidth(120)
            controls_layout.addWidget(self.scan_button)

            # Generate button
            self.generate_button = QPushButton("Generate Model")
            self.generate_button.clicked.connect(self.generate_3d)
            self.generate_button.setEnabled(False)
            self.generate_button.setMinimumWidth(120)
            controls_layout.addWidget(self.generate_button)

            # Save button
            self.save_button = QPushButton("Save Model")
            self.save_button.clicked.connect(self.save_model)
            self.save_button.setEnabled(False)
            self.save_button.setMinimumWidth(120)
            controls_layout.addWidget(self.save_button)

            layout.addLayout(controls_layout)

            # Progress bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setTextVisible(True)
            self.progress_bar.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.progress_bar)

            # Status bar
            self.statusBar().showMessage("Ready")

        except Exception as e:
            print(f"Error setting up UI: {e}")
            QMessageBox.critical(self, "Error", f"Could not set up UI: {str(e)}")
    def create_face_mesh(self, num_vertices=1000):
        """Create more anatomically accurate face mesh template"""
        try:
            # Create a more detailed grid for face topology
            u = np.linspace(-np.pi/2, np.pi/2, 40)
            v = np.linspace(-np.pi/2, np.pi/2, 40)
            u, v = np.meshgrid(u, v)

            # Parameters for face shape
            width = 0.8
            height = 1.0
            depth = 0.6

            # Generate base face shape
            x = width * np.cos(u) * np.cos(v)
            y = height * np.sin(v)
            z = depth * np.cos(v)

            # Add facial features
            # Nose bridge and tip
            nose_bump = 0.2 * np.exp(-((u)**2 + (v+0.2)**2) / 0.1)
            z += nose_bump

            # Cheeks
            cheek_r = 0.1 * np.exp(-((u+0.8)**2 + (v-0.2)**2) / 0.2)
            cheek_l = 0.1 * np.exp(-((u-0.8)**2 + (v-0.2)**2) / 0.2)
            z += cheek_r + cheek_l

            # Flatten back of head
            back_mask = x < -0.2
            z[back_mask] *= 0.7

            # Convert to vertices
            vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
            vertices = torch.tensor(vertices, dtype=torch.float32)

            # Generate faces with better topology
            faces = []
            rows, cols = u.shape
            for i in range(rows - 1):
                for j in range(cols - 1):
                    v0 = i * cols + j
                    v1 = v0 + 1
                    v2 = (i + 1) * cols + j
                    v3 = v2 + 1

                    # Add triangles
                    faces.append([v0, v2, v1])
                    faces.append([v1, v2, v3])

            faces = torch.tensor(faces, dtype=torch.long)
            return vertices, faces

        except Exception as e:
            print(f"Error creating face mesh: {e}")
            return None, None

    def update_mesh_vertices(self, landmarks_3d):
        """Update mesh vertices with improved landmark mapping"""
        try:
            if self.current_mesh is None:
                self.current_mesh = self.template_vertices.cpu().numpy()

            vertices = self.current_mesh.copy()

            # Define key facial feature indices from MediaPipe
            NOSE_TIP = 1
            NOSE_BRIDGE = 168
            LEFT_EYE = 33
            RIGHT_EYE = 263
            LEFT_MOUTH = 61
            RIGHT_MOUTH = 291

            # Create anchor points for deformation
            anchors = {
                'nose_tip': landmarks_3d[NOSE_TIP],
                'nose_bridge': landmarks_3d[NOSE_BRIDGE],
                'left_eye': landmarks_3d[LEFT_EYE],
                'right_eye': landmarks_3d[RIGHT_EYE],
                'left_mouth': landmarks_3d[LEFT_MOUTH],
                'right_mouth': landmarks_3d[RIGHT_MOUTH]
            }

            # Calculate influence weights for each vertex
            weights = np.zeros((len(vertices), len(anchors)))
            for i, vertex in enumerate(vertices):
                for j, (_, anchor) in enumerate(anchors.items()):
                    dist = np.linalg.norm(vertex - anchor)
                    weights[i, j] = 1.0 / (dist + 1e-6)
                weights[i] /= weights[i].sum()  # Normalize weights

            # Apply deformation
            deformed_vertices = np.zeros_like(vertices)
            for i in range(len(vertices)):
                offset = np.zeros(3)
                for j, (_, anchor) in enumerate(anchors.items()):
                    target = anchor - vertices[i]
                    offset += weights[i, j] * target
                deformed_vertices[i] = vertices[i] + offset * 0.5  # Scale factor for smoother deformation

            # Apply temporal smoothing if previous meshes exist
            if len(self.estimated_meshes) > 0:
                alpha = 0.7
                deformed_vertices = alpha * deformed_vertices + (1 - alpha) * self.estimated_meshes[-1]

            return deformed_vertices

        except Exception as e:
            print(f"Error updating mesh vertices: {e}")
            return self.current_mesh

    def process_frame(self, frame, frame_rgb, landmarks):
        """Process frame with improved landmark tracking and mesh updates"""
        try:
            # Convert landmarks to numpy array for easier processing
            landmarks_np = np.array([[l.x, l.y, l.z] for l in landmarks.landmark])

            # Scale and center landmarks
            landmarks_np[:, 0] = (landmarks_np[:, 0] - 0.5) * 2
            landmarks_np[:, 1] = -(landmarks_np[:, 1] - 0.5) * 2
            landmarks_np[:, 2] = -landmarks_np[:, 2] * 3  # Increased depth range

            # Update mesh vertices with improved mapping
            vertices = self.update_mesh_vertices(landmarks_np)

            # Apply enhanced smoothing
            vertices_tensor = torch.tensor(vertices, device=self.device)
            smoothed_vertices = self.smoother.smooth(
                vertices_tensor,
                self.template_faces.to(self.device),
                iterations=3,
                lambda_factor=0.3
            ).cpu().numpy()

            # Update current mesh and store
            self.current_mesh = smoothed_vertices
            self.estimated_meshes.append(smoothed_vertices)
            self.frames.append(frame.copy())

            # Compute colors with improved sampling
            colors = self.compute_vertex_colors(smoothed_vertices, frame_rgb)
            self.gl_widget.update_mesh(
                smoothed_vertices,
                self.template_faces.cpu().numpy(),
                colors
            )

            # Update progress
            if hasattr(self, 'target_frames'):
                progress = min(100, int(len(self.frames) / self.target_frames * 100))
                self.progress_bar.setValue(progress)

                if len(self.frames) >= self.target_frames:
                    self.toggle_scanning()

        except Exception as e:
            print(f"Error processing frame: {e}")
            self.statusBar().showMessage("Error processing frame")

    def update_frame(self):
        """Process frame with improved face tracking and visualization"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                return

            frame = cv2.flip(frame, 1)  # Mirror effect
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                self.draw_face_landmarks(frame, results.multi_face_landmarks[0])

                if self.is_recording:
                    self.process_frame(frame, frame_rgb, results.multi_face_landmarks[0])

            # Display frame
            self.display_frame(frame)

        except Exception as e:
            print(f"Error in frame update: {e}")
            self.statusBar().showMessage("Frame processing error")

    def draw_face_landmarks(self, frame, landmarks):
        """Draw face landmarks with enhanced visualization"""
        try:
            height, width = frame.shape[:2]
            connections = self.mp_face_mesh.FACEMESH_TESSELATION

            # Draw landmarks
            for idx, landmark in enumerate(landmarks.landmark):
                pos = (int(landmark.x * width), int(landmark.y * height))
                cv2.circle(frame, pos, 1, (0, 255, 0), -1)

            # Draw connections with depth-based coloring
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]

                start = landmarks.landmark[start_idx]
                end = landmarks.landmark[end_idx]

                start_pos = (int(start.x * width), int(start.y * height))
                end_pos = (int(end.x * width), int(end.y * height))

                # Color based on depth
                depth = (start.z + end.z) / 2
                color = self.get_depth_color(depth)
                cv2.line(frame, start_pos, end_pos, color, 1)

        except Exception as e:
            print(f"Error drawing landmarks: {e}")

    def get_depth_color(self, depth):
        """Generate color based on depth value"""
        # Convert depth to color (blue to red)
        depth_normalized = (depth + 0.1) / 0.2  # Normalize to 0-1 range
        depth_normalized = np.clip(depth_normalized, 0, 1)

        return (
            int(255 * (1 - depth_normalized)),  # Blue
            0,  # Green
            int(255 * depth_normalized)  # Red
        )

    def process_frame(self, frame, frame_rgb, landmarks):
        """Process frame for 3D reconstruction with improved accuracy"""
        try:
            # Convert landmarks to 3D points
            landmarks_3d = self.landmarks_to_3d(landmarks, frame.shape)

            # Update mesh vertices based on landmarks
            vertices = self.update_mesh_vertices(landmarks_3d)

            # Apply smoothing
            vertices_tensor = torch.tensor(vertices, device=self.device)
            smoothed_vertices = self.smoother.smooth(
                vertices_tensor,
                self.template_faces.to(self.device),
                iterations=1,
                lambda_factor=0.3
            ).cpu().numpy()

            # Update current mesh
            self.current_mesh = smoothed_vertices
            self.estimated_meshes.append(smoothed_vertices)
            self.frames.append(frame.copy())

            # Compute colors and update 3D preview
            colors = self.compute_vertex_colors(smoothed_vertices, frame_rgb)
            self.gl_widget.update_mesh(
                smoothed_vertices,
                self.template_faces.cpu().numpy(),
                colors
            )

            # Update progress
            if hasattr(self, 'target_frames'):
                progress = min(100, int(len(self.frames) / self.target_frames * 100))
                self.progress_bar.setValue(progress)

                if len(self.frames) >= self.target_frames:
                    self.toggle_scanning()

        except Exception as e:
            print(f"Error processing frame: {e}")
            self.statusBar().showMessage("Error processing frame")

    def landmarks_to_3d(self, landmarks, frame_shape):
        """Convert landmarks to 3D points with improved depth estimation"""
        try:
            height, width = frame_shape[:2]
            points_3d = []

            for landmark in landmarks.landmark:
                # Convert to normalized 3D coordinates with depth correction
                x = (landmark.x - 0.5) * 2.0
                y = -(landmark.y - 0.5) * 2.0
                z = -landmark.z * 2.0

                # Apply depth correction based on face topology
                z = z * (1.0 - 0.5 * np.exp(-(x**2 + y**2) / 0.3))

                points_3d.append([x, y, z])

            return np.array(points_3d)

        except Exception as e:
            print(f"Error converting landmarks to 3D: {e}")
            return np.array([])

    def update_mesh_vertices(self, landmarks_3d):
        """Update mesh vertices with improved deformation"""
        try:
            if self.current_mesh is None:
                return self.template_vertices.cpu().numpy()

            vertices = self.current_mesh.copy()

            # Calculate bounding box
            min_bounds = np.min(landmarks_3d, axis=0)
            max_bounds = np.max(landmarks_3d, axis=0)

            # Apply smooth deformation
            vertices_normalized = (vertices - np.min(vertices, axis=0)) / \
                                (np.max(vertices, axis=0) - np.min(vertices, axis=0))
            vertices = vertices_normalized * (max_bounds - min_bounds) + min_bounds

            # Add detail preservation
            if len(self.estimated_meshes) > 0:
                prev_vertices = self.estimated_meshes[-1]
                vertices = 0.7 * vertices + 0.3 * prev_vertices

            return vertices

        except Exception as e:
            print(f"Error updating mesh vertices: {e}")
            return self.current_mesh if self.current_mesh is not None else self.template_vertices.cpu().numpy()

    def compute_vertex_colors(self, vertices, frame):
        """Compute vertex colors with improved texture mapping"""
        try:
            colors = np.ones((len(vertices), 3), dtype=np.uint8) * 180

            # Project vertices to image space
            vertices_homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
            camera_transform = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -2],
                [0, 0, 0, 1]
            ])

            # Transform vertices
            transformed = vertices_homogeneous @ camera_transform.T
            proj_matrix = np.zeros((3, 4))
            proj_matrix[:3, :3] = self.camera_matrix
            projected = (proj_matrix @ transformed.T).T

            # Safe projection with depth handling
            with np.errstate(divide='ignore', invalid='ignore'):
                pixels = np.zeros((len(vertices), 2), dtype=int)
                mask = projected[:, 2] > 0.001
                pixels[mask] = (projected[mask, :2] / projected[mask, 2:3]).astype(int)

            # Valid pixel mask
            valid_mask = (
                (pixels[:, 0] >= 0) & (pixels[:, 0] < frame.shape[1] - 1) &
                (pixels[:, 1] >= 0) & (pixels[:, 1] < frame.shape[0] - 1) &
                mask
            )

            # Bilinear interpolation for better color sampling
            for i in range(len(vertices)):
                if valid_mask[i]:
                    x, y = pixels[i]
                    if 0 <= x < frame.shape[1]-1 and 0 <= y < frame.shape[0]-1:
                        x0, y0 = int(x), int(y)
                        x1, y1 = x0 + 1, y0 + 1
                        wx = x - x0
                        wy = y - y0

                        c00 = frame[y0, x0]
                        c10 = frame[y0, x1]
                        c01 = frame[y1, x0]
                        c11 = frame[y1, x1]

                        color = (1 - wx) * (1 - wy) * c00 + \
                               wx * (1 - wy) * c10 + \
                               (1 - wx) * wy * c01 + \
                               wx * wy * c11

                        colors[i] = color.astype(np.uint8)

            return colors

        except Exception as e:
            print(f"Error computing vertex colors: {e}")
            return np.ones((len(vertices), 3), dtype=np.uint8) * 180
    def toggle_scanning(self):
        """Toggle scanning process with improved frame capture and quality"""
        try:
            if not self.is_recording:
                # Start new scan with significantly more frames
                self.frames = []
                self.estimated_meshes = []
                self.current_mesh = None
                self.progress_bar.setValue(0)
                self.target_frames = 300  # 10 seconds at 30 fps for more complete coverage
                self.scan_button.setText("Stop Scanning")
                self.generate_button.setEnabled(False)
                self.save_button.setEnabled(False)
                self.statusBar().showMessage("Scanning in progress... Rotate your head slowly in all directions (10 seconds)")

                # Initialize pose tracking
                self.last_pose = None
                self.pose_changes = []

                # Reset scan quality metrics
                self.frame_quality = []
                self.last_frame_time = time.time()

            else:
                # Stop scanning with enhanced quality check
                self.scan_button.setText("Start Scan")
                frames_captured = len(self.frames)
                min_required = self.target_frames * 0.7  # More lenient minimum requirement

                if frames_captured > min_required:
                    quality_score = self.assess_scan_quality()
                    pose_coverage = self.calculate_pose_coverage()

                    if quality_score > 0.7 and pose_coverage > 0.6:
                        self.generate_button.setEnabled(True)
                        self.statusBar().showMessage("Scan complete! Good coverage achieved. Generate 3D model when ready.")
                    else:
                        self.statusBar().showMessage("Scan may have insufficient coverage. Consider rescanning with more head rotation.")
                else:
                    self.statusBar().showMessage(f"Scan stopped early ({frames_captured}/{self.target_frames} frames). Please try again.")

            self.is_recording = not self.is_recording

        except Exception as e:
            print(f"Error toggling scan: {e}")
            self.statusBar().showMessage("Error in scanning process")

    def process_frame(self, frame, frame_rgb, landmarks):
        """Process frame with improved landmark tracking and mesh updates"""
        try:
            # Convert landmarks to 3D points with confidence
            landmarks_3d, confidence = self.landmarks_to_3d_with_confidence(landmarks, frame.shape)

            # Skip frames with low confidence
            if confidence < 0.5:
                return

            # Track head pose changes
            current_pose = self.estimate_head_pose(landmarks_3d)
            if self.last_pose is not None:
                pose_change = np.linalg.norm(current_pose - self.last_pose)
                self.pose_changes.append(pose_change)
            self.last_pose = current_pose

            # Update mesh vertices with temporal smoothing
            if len(self.estimated_meshes) > 0:
                prev_vertices = self.estimated_meshes[-1]
                new_vertices = self.update_mesh_vertices(landmarks_3d)
                vertices = self.temporal_smooth(prev_vertices, new_vertices, 0.3)
            else:
                vertices = self.update_mesh_vertices(landmarks_3d)

            # Apply enhanced smoothing
            vertices_tensor = torch.tensor(vertices, device=self.device)
            smoothed_vertices = self.smoother.smooth(
                vertices_tensor,
                self.template_faces.to(self.device),
                iterations=2,  # Increased iterations
                lambda_factor=0.4
            ).cpu().numpy()

            # Update current mesh and store
            self.current_mesh = smoothed_vertices
            self.estimated_meshes.append(smoothed_vertices)
            self.frames.append(frame.copy())

            # Compute colors with improved sampling
            colors = self.compute_vertex_colors_enhanced(smoothed_vertices, frame_rgb)
            self.gl_widget.update_mesh(
                smoothed_vertices,
                self.template_faces.cpu().numpy(),
                colors
            )

            # Update progress
            if hasattr(self, 'target_frames'):
                progress = min(100, int(len(self.frames) / self.target_frames * 100))
                self.progress_bar.setValue(progress)

                if len(self.frames) >= self.target_frames:
                    self.toggle_scanning()

        except Exception as e:
            print(f"Error processing frame: {e}")
            self.statusBar().showMessage("Error processing frame")

    def landmarks_to_3d_with_confidence(self, landmarks, frame_shape):
        """Convert landmarks to 3D points with confidence estimation"""
        height, width = frame_shape[:2]
        points_3d = []
        confidences = []

        for landmark in landmarks.landmark:
            # Convert to normalized 3D coordinates
            x = (landmark.x - 0.5) * 2.0
            y = -(landmark.y - 0.5) * 2.0
            z = -landmark.z * 3.0  # Increased depth range

            # Calculate confidence based on landmark position
            conf = 1.0 - abs(landmark.z)  # Higher confidence for front-facing points
            conf *= (1.0 - min(1.0, abs(x)))  # Lower confidence for extreme side angles
            confidences.append(conf)

            # Apply non-linear depth correction
            z_corrected = z * (1.0 - 0.3 * np.exp(-(x**2 + y**2) / 0.5))
            points_3d.append([x, y, z_corrected])

        points_3d = np.array(points_3d)
        avg_confidence = np.mean(confidences)

        return points_3d, avg_confidence

    def estimate_head_pose(self, landmarks_3d):
        """Estimate head pose from landmarks"""
        # Use key facial landmarks to estimate pose
        if len(landmarks_3d) < 468:  # MediaPipe face mesh has 468 landmarks
            return np.zeros(3)

        # Use eyes and nose bridge points
        left_eye = np.mean(landmarks_3d[33:38], axis=0)
        right_eye = np.mean(landmarks_3d[362:367], axis=0)
        nose = np.mean(landmarks_3d[168:174], axis=0)

        # Calculate pose angles
        forward = np.array([0, 0, 1])
        face_normal = np.cross(right_eye - left_eye, nose - left_eye)
        face_normal = face_normal / np.linalg.norm(face_normal)

        # Calculate rotation angles
        pitch = np.arccos(np.dot(face_normal, forward))
        yaw = np.arctan2(face_normal[0], face_normal[2])
        roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

        return np.array([pitch, yaw, roll])

    def calculate_pose_coverage(self):
        """Calculate how well different head poses are covered"""
        if not self.pose_changes:
            return 0.0

        # Calculate total angular movement
        total_movement = sum(self.pose_changes)

        # Expect at least 2π/3 radians of total movement for good coverage
        coverage = min(1.0, total_movement / (2 * np.pi / 3))

        return coverage

    def temporal_smooth(self, prev_vertices, new_vertices, alpha):
        """Apply temporal smoothing between consecutive frames"""
        return alpha * new_vertices + (1 - alpha) * prev_vertices

    def compute_vertex_colors_enhanced(self, vertices, frame):
        """Compute vertex colors with improved sampling and blending"""
        colors = np.ones((len(vertices), 3), dtype=np.uint8) * 180

        # Project vertices to image space
        vertices_homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)

        # Enhanced camera transform with better perspective
        camera_transform = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -2.5],  # Adjusted camera distance
            [0, 0, 0, 1]
        ])

        # Transform vertices
        transformed = vertices_homogeneous @ camera_transform.T
        proj_matrix = np.zeros((3, 4))
        proj_matrix[:3, :3] = self.camera_matrix
        projected = (proj_matrix @ transformed.T).T

        # Enhanced projection with better depth handling
        with np.errstate(divide='ignore', invalid='ignore'):
            pixels = np.zeros((len(vertices), 2), dtype=int)
            mask = projected[:, 2] > 0.001
            pixels[mask] = (projected[mask, :2] / projected[mask, 2:3]).astype(int)

        # Valid pixel mask with border handling
        valid_mask = (
            (pixels[:, 0] >= 2) & (pixels[:, 0] < frame.shape[1] - 2) &
            (pixels[:, 1] >= 2) & (pixels[:, 1] < frame.shape[0] - 2) &
            mask
        )

        # Enhanced color sampling with larger kernel
        for i in range(len(vertices)):
            if valid_mask[i]:
                x, y = pixels[i]
                # Sample 3x3 neighborhood
                patch = frame[y-1:y+2, x-1:x+2]
                if patch.size > 0:
                    # Weighted average of neighborhood
                    weights = np.array([[1, 2, 1],
                                      [2, 4, 2],
                                      [1, 2, 1]]) / 16.0
                    color = np.sum(patch * weights[:patch.shape[0], :patch.shape[1], np.newaxis], axis=(0,1))
                    colors[i] = color.astype(np.uint8)

        return colors
    def assess_scan_quality(self):
        """Assess the quality of the captured scan"""
        try:
            if not self.frames or not self.estimated_meshes:
                return 0.0

            # Calculate frame coverage
            frame_coverage = len(self.frames) / self.target_frames

            # Calculate face rotation coverage
            landmarks_spread = self.calculate_landmarks_spread()

            # Calculate temporal consistency
            mesh_consistency = self.calculate_mesh_consistency()

            # Weighted quality score
            quality_score = (
                0.4 * frame_coverage +
                0.4 * landmarks_spread +
                0.2 * mesh_consistency
            )

            return min(1.0, quality_score)

        except Exception as e:
            print(f"Error assessing scan quality: {e}")
            return 0.0

    def calculate_landmarks_spread(self):
        """Calculate how well the face rotations cover different angles"""
        try:
            if not self.frames:
                return 0.0

            rotations = []
            for frame in self.frames:
                # Process frame with MediaPipe
                results = self.face_mesh.process(frame)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    # Calculate rough head rotation from landmarks
                    left_eye = np.mean([[l.x, l.y, l.z] for l in landmarks.landmark[33:38]], axis=0)
                    right_eye = np.mean([[l.x, l.y, l.z] for l in landmarks.landmark[362:367]], axis=0)
                    rotation = np.arctan2(right_eye[0] - left_eye[0], right_eye[2] - left_eye[2])
                    rotations.append(rotation)

            if not rotations:
                return 0.0

            # Calculate rotation range coverage
            rotation_range = np.ptp(rotations)
            normalized_coverage = min(1.0, rotation_range / (np.pi/2))  # Expect ±45° rotation

            return normalized_coverage

        except Exception as e:
            print(f"Error calculating landmarks spread: {e}")
            return 0.0

    def calculate_mesh_consistency(self):
        """Calculate consistency between consecutive mesh estimates"""
        try:
            if len(self.estimated_meshes) < 2:
                return 0.0

            diffs = []
            for i in range(1, len(self.estimated_meshes)):
                prev_mesh = self.estimated_meshes[i-1]
                curr_mesh = self.estimated_meshes[i]
                diff = np.mean(np.abs(curr_mesh - prev_mesh))
                diffs.append(diff)

            avg_diff = np.mean(diffs)
            consistency_score = np.exp(-avg_diff * 10)  # Convert to 0-1 score

            return consistency_score

        except Exception as e:
            print(f"Error calculating mesh consistency: {e}")
            return 0.0
    def generate_3d(self):
        """Generate final 3D model with improved mesh fusion"""
        try:
            if len(self.estimated_meshes) < 2:
                raise ValueError("Not enough frames captured")

            self.statusBar().showMessage("Generating 3D model...")
            self.progress_bar.setValue(0)
            QApplication.processEvents()

            # Convert meshes to tensor with robust error handling
            meshes_array = np.stack(self.estimated_meshes)
            meshes_tensor = torch.tensor(meshes_array, device=self.device)

            # Compute weighted average with temporal coherence
            weights = torch.ones(len(self.estimated_meshes), device=self.device)
            weights = torch.softmax(weights, dim=0)  # Normalize weights
            weighted_meshes = meshes_tensor * weights.view(-1, 1, 1)
            final_mesh = weighted_meshes.sum(dim=0)

            self.progress_bar.setValue(50)
            QApplication.processEvents()

            # Apply advanced smoothing
            final_mesh = self.smoother.smooth(
                final_mesh,
                self.template_faces.to(self.device),
                iterations=5,
                lambda_factor=0.5
            )

            # Convert to numpy and normalize
            self.final_mesh = final_mesh.cpu().numpy()
            self.final_mesh = self.normalize_vertices(self.final_mesh)

            # Compute final colors with enhanced sampling
            self.final_colors = self.compute_vertex_colors(self.final_mesh, self.frames[-1])

            # Update 3D preview
            self.gl_widget.update_mesh(
                self.final_mesh,
                self.template_faces.cpu().numpy(),
                self.final_colors
            )

            self.progress_bar.setValue(100)
            self.save_button.setEnabled(True)
            self.statusBar().showMessage("3D model generated successfully!")

        except Exception as e:
            print(f"Error generating 3D model: {e}")
            self.statusBar

def main():
    """Main application entry point with error handling"""
    try:
        app = QApplication(sys.argv)
        window = FaceScanner()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {e}")
        QMessageBox.critical(None, "Error", f"Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()().showMessage("Camera calibration error")
