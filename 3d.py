import sys
import cv2
import os
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QPushButton, QLabel, QProgressBar, QMessageBox, QHBoxLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class HeadScannerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Head Scanner")
        self.frames = []
        self.keypoints_list = []
        self.descriptors_list = []
        self.face_landmarks_list = []

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Camera parameters (approximate values for webcam)
        self.focal_length = 1000
        self.camera_matrix = np.array([
            [self.focal_length, 0, 1280/2],
            [0, self.focal_length, 720/2],
            [0, 0, 1]
        ])

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create info label
        self.info_label = QLabel(
            "Instructions:\n"
            "1. Sit about 60cm from camera\n"
            "2. Slowly rotate your head 360° (45° at a time)\n"
            "3. Keep your face centered in frame\n"
            "4. Try to maintain neutral expression"
        )
        self.info_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(self.info_label)

        # Create camera display label
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # Create button layout
        button_layout = QHBoxLayout()

        # Create buttons
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        button_layout.addWidget(self.record_button)

        self.generate_button = QPushButton("Generate 3D Model")
        self.generate_button.clicked.connect(self.generate_3d)
        self.generate_button.setEnabled(False)
        button_layout.addWidget(self.generate_button)

        layout.addLayout(button_layout)

        # Create progress and status layout
        status_layout = QVBoxLayout()

        # Create progress bar
        self.progress = QProgressBar()
        status_layout.addWidget(self.progress)

        # Create status label
        self.status_label = QLabel("Ready to scan. Position your head in the frame.")
        self.status_label.setStyleSheet("color: #333; font-weight: bold;")
        status_layout.addWidget(self.status_label)

        layout.addLayout(status_layout)

        # Recording state
        self.is_recording = False
        self.max_frames = 200  # Increased to 200 frames for better coverage

        # Initialize camera
        print("Opening camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("Error: Could not open camera!")
            return

        # Set higher resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Feature detector with optimized parameters for faces
        self.detector = cv2.SIFT_create(
            nfeatures=3000,
            nOctaveLayers=5,
            contrastThreshold=0.02,
            edgeThreshold=15,
            sigma=2.0
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Set up timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms

        # Set window size
        self.setMinimumSize(1000, 800)
        print("App initialized")

    def process_face_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        landmarks = []

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]

            # Draw face mesh
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        return frame, landmarks

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                # Process face landmarks and draw mesh
                frame_with_mesh, landmarks = self.process_face_landmarks(frame.copy())

                if self.is_recording and len(self.frames) < self.max_frames:
                    if landmarks:  # Only store frame if face is detected
                        # Detect features
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

                        if keypoints and len(keypoints) > 30:
                            # Store frame and features
                            self.frames.append(frame.copy())
                            self.keypoints_list.append(keypoints)
                            self.descriptors_list.append(descriptors)
                            self.face_landmarks_list.append(landmarks)

                            # Update progress
                            progress = int(len(self.frames) / self.max_frames * 100)
                            self.progress.setValue(progress)

                            # Update status with rotation guidance
                            rotation_step = (len(self.frames) // 25) * 45
                            self.status_label.setText(f"Recording... {len(self.frames)}/{self.max_frames} frames. Rotate to approximately {rotation_step}°")

                            if len(self.frames) >= self.max_frames:
                                self.toggle_recording()
                        else:
                            self.status_label.setText("Not enough facial features detected. Adjust lighting or position.")
                    else:
                        self.status_label.setText("No face detected! Please center your face in the frame.")

                # Convert frame to RGB for display
                rgb_frame = cv2.cvtColor(frame_with_mesh, cv2.COLOR_BGR2RGB)

                # Convert to QImage
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # Scale to fit label
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

                # Display the image
                self.image_label.setPixmap(scaled_pixmap)

            else:
                self.status_label.setText("Failed to capture frame")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            print(f"Error updating frame: {str(e)}")

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.frames = []
            self.keypoints_list = []
            self.descriptors_list = []
            self.face_landmarks_list = []
            self.record_button.setText("Stop Recording")
            self.status_label.setText("Recording... Center your face and slowly rotate.")
            self.generate_button.setEnabled(False)
        else:
            self.record_button.setText("Start Recording")
            self.status_label.setText("Recording stopped")
            if len(self.frames) >= 50:
                self.generate_button.setEnabled(True)
            else:
                self.status_label.setText("Not enough frames captured. Try again.")

    def generate_3d(self):
        try:
            if len(self.frames) < 50:
                QMessageBox.warning(self, "Warning", "Not enough frames captured")
                return

            self.status_label.setText("Generating 3D structure...")
            self.progress.setValue(0)

            # Combine face landmarks and SIFT features for better reconstruction
            points3D = []
            colors = []
            valid_points = 0

            # Process frames in pairs with overlap
            for i in range(0, len(self.frames)-1):
                # Match features
                matches = self.matcher.knnMatch(self.descriptors_list[i],
                                             self.descriptors_list[i+1], k=2)

                # Lowe's ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

                if len(good_matches) < 50:
                    continue

                # Get matched points
                pts1 = np.float32([self.keypoints_list[i][m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([self.keypoints_list[i+1][m.trainIdx].pt for m in good_matches])

                # Calculate essential matrix
                E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix,
                                             method=cv2.RANSAC,
                                             prob=0.999,
                                             threshold=1.0)

                if E is None:
                    continue

                # Filter points using mask
                pts1 = pts1[mask.ravel() == 1]
                pts2 = pts2[mask.ravel() == 1]

                # Recover pose
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)

                # Create projection matrices
                P1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
                P2 = self.camera_matrix @ np.hstack((R, t))

                # Triangulate points
                points4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
                points4D /= points4D[3]

                # Add face landmarks to 3D points
                landmarks1 = np.array(self.face_landmarks_list[i])
                landmarks2 = np.array(self.face_landmarks_list[i+1])

                # Triangulate face landmarks
                lm_pts1 = np.float32([[lm[0] * self.frames[i].shape[1],
                                     lm[1] * self.frames[i].shape[0]] for lm in landmarks1])
                lm_pts2 = np.float32([[lm[0] * self.frames[i+1].shape[1],
                                     lm[1] * self.frames[i+1].shape[0]] for lm in landmarks2])

                lm_points4D = cv2.triangulatePoints(P1, P2, lm_pts1.T, lm_pts2.T)
                lm_points4D /= lm_points4D[3]

                # Combine feature points and landmarks
                all_points4D = np.concatenate([points4D, lm_points4D], axis=1)
                all_pts1 = np.concatenate([pts1, lm_pts1])

                # Filter valid points
                valid_mask = np.abs(all_points4D[3]) > 1e-8
                all_points4D = all_points4D[:, valid_mask]
                all_pts1 = all_pts1[valid_mask]

                # Add valid points and their colors
                for j, pt4D in enumerate(all_points4D.T):
                    if np.all(np.abs(pt4D[:3]) < 100):  # Basic outlier filtering
                        points3D.append(pt4D[:3])
                        x, y = int(all_pts1[j][0]), int(all_pts1[j][1])
                        if 0 <= x < self.frames[i].shape[1] and 0 <= y < self.frames[i].shape[0]:
                            colors.append(self.frames[i][y,x])
                            valid_points += 1

                self.progress.setValue(int((i+1)/(len(self.frames)-1) * 100))

            if valid_points < 1000:  # Increased minimum point threshold
                raise Exception("Not enough valid 3D points reconstructed")

            print(f"Reconstructed {valid_points} valid 3D points")
            self.save_ply(points3D, colors, "head_reconstruction.ply")

            self.status_label.setText("3D reconstruction saved as head_reconstruction.ply")
            QMessageBox.information(self, "Success",
                                  f"3D head reconstruction completed with {valid_points} points")

        except Exception as e:
            self.status_label.setText(f"Error in 3D reconstruction: {str(e)}")
            print(f"Error in generate_3d: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to generate 3D model: {str(e)}")

    def save_ply(self, points, colors, filename):
        if not points:
            raise Exception("No points to save")

        points = np.array(points)
        colors = np.array(colors)

        # Filter out invalid points
        valid_mask = ~np.any(np.isnan(points), axis=1)
        points = points[valid_mask]
        colors = colors[:len(points)] if len(colors) > len(points) else colors

        print(f"Saving {len(points)} points to PLY file")

        with open(filename, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # Write vertex list with high precision
            for i, point in enumerate(points):
                if i < len(colors):
                    color = colors[i]
                    f.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                           f"{int(color[2])} {int(color[1])} {int(color[0])}\n")
                else:
                    f.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} 255 255 255\n")

    def closeEvent(self, event):
        print("Closing application...")
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        self.face_mesh.close()
        event.accept()

def main():
    print("Starting application...")
    app = QApplication(sys.argv)
    window = HeadScannerApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()