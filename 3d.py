import sys
import cv2
import os
import time
import numpy as np
import torch
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QProgressBar,
                           QGroupBox, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial import Delaunay

class GLMeshViewer(QGLWidget):
    def __init__(self, parent=None):
        super(GLMeshViewer, self).__init__(parent)
        self.mesh_vertices = None
        self.mesh_faces = None
        self.mesh_colors = None
        self.rotation = [0.0, 0.0, 0.0]
        self.translation = [0.0, 0.0, -5.0]
        self.scale = 1.0

        # Start rotation timer
        self.rotation_timer = QTimer(self)
        self.rotation_timer.timeout.connect(self.rotate)
        self.rotation_timer.start(50)

    def rotate(self):
        self.rotation[1] += 2.0
        self.updateGL()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])

        # Set material properties
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)

        glClearColor(0.8, 0.8, 0.8, 1.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Set up camera
        glTranslatef(*self.translation)
        glRotatef(self.rotation[0], 1.0, 0.0, 0.0)
        glRotatef(self.rotation[1], 0.0, 1.0, 0.0)
        glRotatef(self.rotation[2], 0.0, 0.0, 1.0)
        glScalef(self.scale, self.scale, self.scale)

        if self.mesh_vertices is not None and self.mesh_faces is not None:
            # Draw solid mesh
            glEnable(GL_LIGHTING)
            glBegin(GL_TRIANGLES)
            for face in self.mesh_faces:
                # Calculate face normal for better lighting
                v0 = self.mesh_vertices[face[0]]
                v1 = self.mesh_vertices[face[1]]
                v2 = self.mesh_vertices[face[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                normal = normal / np.linalg.norm(normal)
                glNormal3fv(normal)

                for idx in face:
                    if idx < len(self.mesh_vertices):
                        if self.mesh_colors is not None:
                            color = self.mesh_colors[idx]
                            glColor3f(color[0]/255.0, color[1]/255.0, color[2]/255.0)
                        else:
                            glColor3f(0.8, 0.8, 0.8)
                        vertex = self.mesh_vertices[idx]
                        glVertex3f(*vertex)
            glEnd()

            # Draw wireframe overlay
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

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(w)/float(h), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def update_mesh(self, vertices, faces, colors=None):
        if vertices is not None:
            # Center and scale the mesh
            center = np.mean(vertices, axis=0)
            vertices = vertices - center
            scale = np.max(np.abs(vertices))
            if scale > 0:
                vertices = vertices / scale

            self.mesh_vertices = vertices
            self.mesh_faces = faces
            self.mesh_colors = colors
            self.updateGL()

class FaceScanner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Face Scanner")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize state variables
        self.is_recording = False
        self.frames = []
        self.landmarks_sequence = []
        self.current_mesh = None
        self.final_mesh = None
        self.final_colors = None

        # Setup UI and components
        self.setup_ui()
        self.setup_face_tracking()
        self.initialize_camera()

    def setup_face_tracking(self):
        """Initialize MediaPipe face mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )

    def setup_ui(self):
        """Set up the user interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Preview layout
        preview_layout = QHBoxLayout()

        # Camera feed
        camera_group = QGroupBox("Camera Feed")
        camera_layout = QVBoxLayout(camera_group)
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        camera_layout.addWidget(self.image_label)
        preview_layout.addWidget(camera_group)

        # 3D preview
        model_group = QGroupBox("3D Model Preview")
        model_layout = QVBoxLayout(model_group)
        self.gl_widget = GLMeshViewer()
        model_layout.addWidget(self.gl_widget)
        preview_layout.addWidget(model_group)

        layout.addLayout(preview_layout)

        # Controls
        controls_layout = QHBoxLayout()

        self.scan_button = QPushButton("Start Scan")
        self.scan_button.clicked.connect(self.toggle_scanning)
        controls_layout.addWidget(self.scan_button)

        self.generate_button = QPushButton("Generate Model")
        self.generate_button.clicked.connect(self.generate_3d)
        self.generate_button.setEnabled(False)
        controls_layout.addWidget(self.generate_button)

        self.save_button = QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        controls_layout.addWidget(self.save_button)

        layout.addLayout(controls_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Status bar
        self.statusBar().showMessage("Ready")

    def initialize_camera(self):
        """Initialize webcam capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open camera")
            return False

        # Set up frame timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS
        return True

    def update_frame(self):
        """Process each frame from the camera"""
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  # Mirror
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]

        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # Convert landmarks to 3D points with proper scaling
            landmarks_array = []
            for landmark in landmarks.landmark:
                # Scale x and y to pixel coordinates
                x = landmark.x * width
                y = landmark.y * height
                # Scale z relative to face width
                z = landmark.z * width
                landmarks_array.append([x, y, z])

            landmarks_array = np.array(landmarks_array)

            # Draw face mesh visualization
            self.draw_face_mesh(frame, landmarks_array)

            if self.is_recording:
                self.landmarks_sequence.append(landmarks_array)
                self.frames.append(frame.copy())

                # Update progress bar
                if len(self.landmarks_sequence) >= 30:  # 1 second of frames
                    self.toggle_scanning()
                else:
                    progress = int((len(self.landmarks_sequence) / 30) * 100)
                    self.progress_bar.setValue(progress)

        # Display frame
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio))

    def draw_face_mesh(self, frame, landmarks):
        """Draw improved face mesh visualization"""
        # Draw tessellation
        connections = self.mp_face_mesh.FACEMESH_TESSELATION

        # Calculate depth range for better coloring
        z_values = landmarks[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        z_range = z_max - z_min if z_max > z_min else 1.0

        # Draw edges with depth-based coloring
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]

            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx][:2].astype(int)
                end_point = landmarks[end_idx][:2].astype(int)

                # Color based on average depth
                z_avg = (landmarks[start_idx][2] + landmarks[end_idx][2]) / 2
                depth_value = (z_avg - z_min) / z_range
                color = (
                    int(255 * (1 - depth_value)),  # Blue
                    int(255 * depth_value),        # Green
                    0                              # Red
                )

                cv2.line(frame, tuple(start_point), tuple(end_point), color, 1)

        # Draw landmarks
        for landmark in landmarks:
            x, y, z = landmark
            x, y = int(x), int(y)
            depth_value = (z - z_min) / z_range
            color = (
                int(255 * (1 - depth_value)),
                int(255 * depth_value),
                0
            )
            cv2.circle(frame, (x, y), 2, color, -1)

    def toggle_scanning(self):
        """Start or stop scanning"""
        self.is_recording = not self.is_recording

        if self.is_recording:
            self.landmarks_sequence = []
            self.frames = []
            self.scan_button.setText("Stop Scanning")
            self.generate_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("Recording... Move your head slowly")
        else:
            self.scan_button.setText("Start Scan")
            if len(self.landmarks_sequence) > 0:
                self.generate_button.setEnabled(True)
                self.statusBar().showMessage("Scan complete. Generate 3D model.")

    def generate_3d(self):
        """Generate improved 3D model from captured landmarks"""
        if not self.landmarks_sequence:
            return

        try:
            self.statusBar().showMessage("Generating 3D model...")
            self.progress_bar.setValue(0)

            # Calculate mean landmarks for stable mesh
            mean_landmarks = np.mean(self.landmarks_sequence, axis=0)

            # Create vertex positions from landmarks
            vertices = mean_landmarks.copy()

            # Center the mesh
            center = np.mean(vertices, axis=0)
            vertices = vertices - center

            # Scale to reasonable size
            scale = 1.0 / np.max(np.abs(vertices))
            vertices = vertices * scale

            # Create faces using Delaunay triangulation
            # Project points to 2D for triangulation
            points_2d = vertices[:, :2]
            tri = Delaunay(points_2d)
            faces = tri.simplices

            # Store final mesh
            self.final_mesh = vertices
            self.final_colors = np.ones((len(vertices), 3), dtype=np.uint8) * 180

            # Update 3D preview
            self.gl_widget.update_mesh(
                vertices,
                faces,
                self.final_colors
            )

            self.save_button.setEnabled(True)
            self.progress_bar.setValue(100)
            self.statusBar().showMessage("3D model generated successfully!")

        except Exception as e:
            print(f"Error generating 3D model: {e}")
            self.statusBar().showMessage("Error generating 3D model")
            QMessageBox.critical(self, "Error", f"Failed to generate 3D model: {str(e)}")

    def filter_triangles(self, vertices, faces):
        """Filter out problematic triangles"""
        valid_faces = []
        for face in faces:
            # Get triangle vertices
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            # Calculate edge lengths
            edge1 = np.linalg.norm(v1 - v0)
            edge2 = np.linalg.norm(v2 - v1)
            edge3 = np.linalg.norm(v0 - v2)

            # Calculate triangle area
            s = (edge1 + edge2 + edge3) / 2
            area = np.sqrt(s * (s - edge1) * (s - edge2) * (s - edge3))

            # Filter based on edge lengths and area
            max_edge = max(edge1, edge2, edge3)
            if area > 0.0001 and max_edge < 0.5:
                valid_faces.append(face)

        return np.array(valid_faces)

    def save_model(self):
        """Save the 3D model to file"""
        if self.final_mesh is None:
            return

        try:
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Save 3D Model",
                "face_scan.obj",
                "OBJ files (*.obj);;PLY files (*.ply)"
            )

            if not file_path:
                return

            self.progress_bar.setValue(0)
            self.statusBar().showMessage("Saving 3D model...")

            if file_path.endswith('.obj'):
                self.save_as_obj(file_path)
            elif file_path.endswith('.ply'):
                self.save_as_ply(file_path)
            else:
                file_path += '.obj'
                self.save_as_obj(file_path)

            self.progress_bar.setValue(100)
            self.statusBar().showMessage("Model saved successfully!")

        except Exception as e:
            print(f"Error saving model: {e}")
            self.statusBar().showMessage("Error saving model")
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")

    def save_as_obj(self, file_path):
        """Save model in OBJ format with materials"""
        # Create material file
        mtl_path = file_path.replace('.obj', '.mtl')
        with open(mtl_path, 'w') as f:
            f.write("newmtl FaceMaterial\n")
            f.write("Ka 0.2 0.2 0.2\n")  # Ambient color
            f.write("Kd 0.8 0.8 0.8\n")  # Diffuse color
            f.write("Ks 0.5 0.5 0.5\n")  # Specular color
            f.write("Ns 50.0\n")         # Shininess
            f.write("d 1.0\n")           # Opacity

        # Save OBJ file
        with open(file_path, 'w') as f:
            f.write(f"mtllib {os.path.basename(mtl_path)}\n")

            # Write vertices
            for vertex in self.final_mesh:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

            # Calculate and write vertex normals
            vertex_normals = self.calculate_vertex_normals()
            for normal in vertex_normals:
                f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")

            f.write("usemtl FaceMaterial\n")

            # Write faces with vertex normals
            for i, face in enumerate(self.gl_widget.mesh_faces):
                # OBJ indices are 1-based
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} "
                       f"{face[2]+1}//{face[2]+1}\n")

    def save_as_ply(self, file_path):
        """Save model in PLY format with vertex colors"""
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
            f.write(f"element face {len(self.gl_widget.mesh_faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Write vertices with colors
            for i, vertex in enumerate(self.final_mesh):
                color = self.final_colors[i]
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} "
                       f"{color[0]} {color[1]} {color[2]}\n")

            # Write faces
            for face in self.gl_widget.mesh_faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    def calculate_vertex_normals(self):
        """Calculate vertex normals for smooth shading"""
        vertices = self.final_mesh
        faces = self.gl_widget.mesh_faces

        # Initialize normal array
        vertex_normals = np.zeros_like(vertices)

        # Calculate face normals and accumulate to vertices
        for face in faces:
            v0, v1, v2 = vertices[face]
            # Calculate face normal
            normal = np.cross(v1 - v0, v2 - v0)
            # Add to all vertices of this face
            vertex_normals[face] += normal

        # Normalize the normals
        norms = np.linalg.norm(vertex_normals, axis=1)
        vertex_normals[norms > 0] /= norms[norms > 0, np.newaxis]

        return vertex_normals

    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        try:
            # Stop timers
            self.timer.stop()
            self.gl_widget.rotation_timer.stop()

            # Release camera
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()

            # Clean up MediaPipe
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()

            event.accept()

        except Exception as e:
            print(f"Error during cleanup: {e}")
            event.accept()

def main():
    """Main application entry point"""
    try:
        app = QApplication(sys.argv)

        # Set application style
        app.setStyle('Fusion')

        # Create and show main window
        window = FaceScanner()
        window.show()

        sys.exit(app.exec_())

    except Exception as e:
        print(f"Application error: {e}")
        if 'app' in locals():
            QMessageBox.critical(None, "Error", f"Application failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()