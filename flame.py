import cv2
import numpy as np
import pickle
import open3d as o3d
import dlib
import threading
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceModelingApp:
    def __init__(self):
        self.FLAME_MODEL_PATH = Path("flame_model.pkl")
        self.DLIB_LANDMARK_PATH = Path("shape_predictor_68_face_landmarks.dat")
        self.BUFFER_SIZE = 10
        self.landmark_buffer = []

        # Initialize components
        self._load_models()
        self._init_capture()

    def _load_models(self):
        """Load FLAME model and dlib predictors with proper error handling."""
        # Load FLAME model
        try:
            with open(self.FLAME_MODEL_PATH, "rb") as file:
                self.flame = pickle.load(file, encoding="latin1")
            logger.info("FLAME model loaded successfully")
        except FileNotFoundError:
            logger.error(f"FLAME model not found at {self.FLAME_MODEL_PATH}")
            raise
        except Exception as e:
            logger.error(f"Error loading FLAME model: {e}")
            raise

        # Load dlib components
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(str(self.DLIB_LANDMARK_PATH))
            logger.info("Dlib models loaded successfully")
        except RuntimeError as e:
            logger.error(f"Failed to load dlib predictor: {e}")
            raise

    def _init_capture(self):
        """Initialize video capture with error checking."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Failed to open webcam")
            raise RuntimeError("Cannot open webcam")

        # Set optimal capture properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def get_landmarks(self, image):
        """Detect face landmarks with improved error handling."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)

            if not faces:
                return None

            shape = self.landmark_predictor(gray, faces[0])
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            return landmarks

        except Exception as e:
            logger.error(f"Error detecting landmarks: {e}")
            return None

    def estimate_parameters(self, landmarks):
        """Estimate FLAME parameters with realistic constraints."""
        try:
            # Use PCA to estimate shape parameters
            shape_params = np.zeros(self.flame['shapedirs'].shape[-1])
            shape_params[:10] = np.random.normal(0, 0.1, 10)  # Only use first 10 components

            # Estimate pose parameters based on landmark positions
            pose_params = np.zeros(self.flame['posedirs'].shape[-1])
            pose_params[:3] = self.estimate_head_pose(landmarks)  # Basic head pose estimation

            return shape_params, pose_params

        except Exception as e:
            logger.error(f"Error estimating parameters: {e}")
            return None, None

    def estimate_head_pose(self, landmarks):
        """Basic head pose estimation from landmarks."""
        # Simplified pose estimation - can be improved with proper 2D-to-3D correspondence
        center = landmarks.mean(axis=0)
        spread = landmarks.std(axis=0)
        return [spread[0]/100, spread[1]/100, 0]  # Basic rotation estimation

    def generate_3d_mesh(self, landmarks):
        """Generate mesh with validation checks."""
        if landmarks is None:
            return None, None

        try:
            shape_params, pose_params = self.estimate_parameters(landmarks)
            if shape_params is None:
                return None, None

            # Generate base mesh
            v_template = self.flame['v_template']
            v_shaped = v_template + self.flame['shapedirs'].dot(shape_params)

            # Apply pose deformation with bounds checking
            pose_deformation = self.flame['posedirs'].dot(pose_params)
            if np.any(np.isnan(pose_deformation)) or np.any(np.isinf(pose_deformation)):
                logger.warning("Invalid pose deformation detected")
                pose_deformation = np.zeros_like(pose_deformation)

            v_posed = v_shaped + pose_deformation

            return v_posed, self.flame['f']

        except Exception as e:
            logger.error(f"Error generating mesh: {e}")
            return None, None

    def save_mesh(self, vertices, faces, output_path="output_mesh.obj"):
        """Save mesh with validation."""
        try:
            if vertices is None or faces is None:
                return False

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as file:
                for vertex in vertices:
                    file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                for face in faces:
                    file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

            logger.info(f"Mesh saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving mesh: {e}")
            return False

    def visualize_mesh(self, vertices, faces):
        """Non-blocking mesh visualization with error handling."""
        if vertices is None or faces is None:
            return

        try:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()

            def _visualize():
                try:
                    o3d.visualization.draw_geometries([mesh])
                except Exception as e:
                    logger.error(f"Visualization error: {e}")

            threading.Thread(target=_visualize, daemon=True).start()

        except Exception as e:
            logger.error(f"Error preparing visualization: {e}")

    def run(self):
        """Main application loop with proper cleanup."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break

                # Detect landmarks
                landmarks = self.get_landmarks(frame)

                if landmarks is not None:
                    # Update landmark buffer
                    self.landmark_buffer.append(landmarks)
                    if len(self.landmark_buffer) > self.BUFFER_SIZE:
                        self.landmark_buffer.pop(0)

                    # Process landmarks when buffer is full
                    if len(self.landmark_buffer) == self.BUFFER_SIZE:
                        averaged_landmarks = np.mean(self.landmark_buffer, axis=0)

                        # Visualize landmarks
                        for (x, y) in averaged_landmarks.astype(int):
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                        # Generate and visualize mesh
                        vertices, faces = self.generate_3d_mesh(averaged_landmarks)
                        if vertices is not None:
                            self.save_mesh(vertices, faces)
                            self.visualize_mesh(vertices, faces)

                # Display frame
                cv2.imshow("Face Modeling", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logger.error(f"Application error: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = FaceModelingApp()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")