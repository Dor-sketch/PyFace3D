import cv2
import numpy as np
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import threading
from queue import Queue
import time
from scipy.spatial import Delaunay

class FaceModelGenerator:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize display
        pygame.init()
        self.display_width = 1200
        self.display_height = 600
        pygame.display.set_mode((self.display_width, self.display_height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Face Model Generator")

        # OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)

        # Set up camera
        gluPerspective(45, (self.display_width/2) / self.display_height, 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)

        # Initialize face tracking
        self.points_buffer = []
        self.max_buffer_size = 30
        self.mesh_data = None
        self.rotation = [0, 0, 0]

        # Threading setup
        self.running = True
        self.mesh_queue = Queue(maxsize=1)

    def process_frame(self, frame):
        """Process webcam frame and extract face landmarks"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        # Extract 3D points
        points = []
        for landmark in results.multi_face_landmarks[0].landmark:
            points.append([landmark.x - 0.5, landmark.y - 0.5, landmark.z])

        return np.array(points)

    def create_mesh(self, points):
        """Create triangulated mesh from points"""
        # Project points to 2D for triangulation
        points_2d = points[:, :2]
        tri = Delaunay(points_2d)

        return {
            'vertices': points,
            'triangles': tri.simplices,
            'normals': self.calculate_normals(points, tri.simplices)
        }

    def calculate_normals(self, vertices, triangles):
        """Calculate surface normals for smooth shading"""
        normals = np.zeros_like(vertices)

        for triangle in triangles:
            v1 = vertices[triangle[1]] - vertices[triangle[0]]
            v2 = vertices[triangle[2]] - vertices[triangle[0]]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

            normals[triangle[0]] += normal
            normals[triangle[1]] += normal
            normals[triangle[2]] += normal

        # Normalize
        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1
        normals = normals / norms[:, np.newaxis]

        return normals

    def draw_mesh(self, mesh_data):
        """Draw the 3D mesh using OpenGL"""
        glPushMatrix()

        # Apply rotation
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)

        # Draw triangles
        glBegin(GL_TRIANGLES)
        glColor3f(0.7, 0.7, 1.0)

        for triangle in mesh_data['triangles']:
            for vertex_id in triangle:
                normal = mesh_data['normals'][vertex_id]
                vertex = mesh_data['vertices'][vertex_id]
                glNormal3fv(normal)
                glVertex3fv(vertex)

        glEnd()
        glPopMatrix()

    def draw_frame(self, frame):
        """Draw webcam frame using OpenGL texture"""
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 0)

        # Create texture
        texture_surface = pygame.surfarray.make_surface(frame_rgb)
        texture_data = pygame.image.tostring(texture_surface, 'RGB', 1)

        # Set up texture
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.shape[1], frame.shape[0], 0,
                    GL_RGB, GL_UNSIGNED_BYTE, texture_data)

        # Draw textured quad
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(-1, 1)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        # Clean up
        glDeleteTextures([texture_id])

    def run(self):
        cap = cv2.VideoCapture(0)
        clock = pygame.time.Clock()

        try:
            while self.running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_s:
                            # Save model
                            if self.mesh_data:
                                self.save_model()
                        elif event.key == pygame.K_r:
                            # Reset rotation
                            self.rotation = [0, 0, 0]
                    elif event.type == pygame.MOUSEMOTION:
                        if event.buttons[0]:  # Left mouse button
                            self.rotation[1] += event.rel[0]
                            self.rotation[0] += event.rel[1]

                # Process webcam frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Get face landmarks
                points = self.process_frame(frame)

                if points is not None:
                    # Update points buffer
                    self.points_buffer.append(points)
                    if len(self.points_buffer) > self.max_buffer_size:
                        self.points_buffer.pop(0)

                    # Average points for stability
                    avg_points = np.mean(self.points_buffer, axis=0)

                    # Create mesh
                    self.mesh_data = self.create_mesh(avg_points)

                # Clear screen
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # Set up split screen
                glViewport(0, 0, self.display_width // 2, self.display_height)
                self.draw_frame(frame)

                glViewport(self.display_width // 2, 0,
                          self.display_width // 2, self.display_height)
                if self.mesh_data:
                    self.draw_mesh(self.mesh_data)

                # Update display
                pygame.display.flip()
                clock.tick(60)

        finally:
            cap.release()
            pygame.quit()
            self.face_mesh.close()

    def save_model(self):
        """Save the current model to OBJ file"""
        if not self.mesh_data:
            return

        filename = f"face_model_{int(time.time())}.obj"
        with open(filename, 'w') as f:
            # Write vertices
            for vertex in self.mesh_data['vertices']:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            # Write normals
            for normal in self.mesh_data['normals']:
                f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

            # Write faces
            for i, triangle in enumerate(self.mesh_data['triangles']):
                f.write(f"f {triangle[0]+1}//{triangle[0]+1} "
                       f"{triangle[1]+1}//{triangle[1]+1} "
                       f"{triangle[2]+1}//{triangle[2]+1}\n")

        print(f"Model saved as {filename}")

if __name__ == "__main__":
    generator = FaceModelGenerator()
    generator.run()