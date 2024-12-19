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
        self.display_width = 1400
        self.display_height = 800
        self.window_surface = pygame.display.set_mode(
            (self.display_width, self.display_height),
            DOUBLEBUF | OPENGL | RESIZABLE
        )
        pygame.display.set_caption("Textured Face Model Generator")

        # OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 1, 0])
        glEnable(GL_COLOR_MATERIAL)

        # Set up camera view
        gluPerspective(45, (self.display_width/2) / self.display_height, 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)

        # Initialize tracking
        self.points_buffer = []
        self.max_buffer_size = 30
        self.mesh_data = None
        self.rotation = [0, 0, 0]
        self.running = True  # Added running flag
        self.texture_id = None
        self.last_frame = None

    def process_frame(self, frame):
        """Process webcam frame and extract face landmarks"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.last_frame = frame_rgb  # Store for texture mapping
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

        # Calculate texture coordinates
        texture_coords = points_2d.copy()
        texture_coords[:, 0] = (texture_coords[:, 0] + 0.5)
        texture_coords[:, 1] = 1.0 - (texture_coords[:, 1] + 0.5)

        return {
            'vertices': points,
            'triangles': tri.simplices,
            'normals': self.calculate_normals(points, tri.simplices),
            'texture_coords': texture_coords
        }

    def calculate_normals(self, vertices, triangles):
        """Calculate surface normals for lighting"""
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

    def update_texture(self, frame):
        """Update OpenGL texture with current frame"""
        if frame is None:
            return

        # Flip frame vertically
        frame = cv2.flip(frame, 0)

        # Convert to string buffer
        texture_data = pygame.image.tostring(
            pygame.surfarray.make_surface(frame), 'RGB', 1)

        # Generate texture if needed
        if self.texture_id is None:
            self.texture_id = glGenTextures(1)

        # Update texture
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.shape[1], frame.shape[0],
                    0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)

    def draw_mesh(self, mesh_data):
        """Draw the textured 3D mesh"""
        if self.last_frame is not None:
            self.update_texture(self.last_frame)

        glPushMatrix()
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        glBegin(GL_TRIANGLES)
        for triangle in mesh_data['triangles']:
            for vertex_id in triangle:
                normal = mesh_data['normals'][vertex_id]
                vertex = mesh_data['vertices'][vertex_id]
                texcoord = mesh_data['texture_coords'][vertex_id]

                glNormal3fv(normal)
                glTexCoord2fv(texcoord)
                glVertex3fv(vertex)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glPopMatrix()

    def draw_camera_feed(self, frame):
        """Draw the camera feed on the left side"""
        if frame is None:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 0)

        texture_surface = pygame.surfarray.make_surface(frame_rgb)
        texture_data = pygame.image.tostring(texture_surface, 'RGB', 1)

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.shape[1], frame.shape[0],
                    0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)

        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(-1, 1)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        glDeleteTextures([texture_id])

    def save_model(self):
        """Save the current model to OBJ file with texture"""
        if not self.mesh_data or self.last_frame is None:
            return

        timestamp = int(time.time())
        obj_filename = f"face_model_{timestamp}.obj"
        texture_filename = f"face_texture_{timestamp}.png"

        # Save texture
        cv2.imwrite(texture_filename, cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2BGR))

        # Write OBJ file
        with open(obj_filename, 'w') as f:
            f.write(f"mtllib {obj_filename}.mtl\n")

            # Write vertices
            for vertex in self.mesh_data['vertices']:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            # Write texture coordinates
            for texcoord in self.mesh_data['texture_coords']:
                f.write(f"vt {texcoord[0]} {texcoord[1]}\n")

            # Write normals
            for normal in self.mesh_data['normals']:
                f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

            f.write("usemtl face_material\n")
            # Write faces with texture coordinates and normals
            for triangle in self.mesh_data['triangles']:
                f.write(f"f {triangle[0]+1}/{triangle[0]+1}/{triangle[0]+1} "
                       f"{triangle[1]+1}/{triangle[1]+1}/{triangle[1]+1} "
                       f"{triangle[2]+1}/{triangle[2]+1}/{triangle[2]+1}\n")

        # Write MTL file
        with open(f"{obj_filename}.mtl", 'w') as f:
            f.write("newmtl face_material\n")
            f.write("Ka 1.000 1.000 1.000\n")
            f.write("Kd 1.000 1.000 1.000\n")
            f.write("Ks 0.000 0.000 0.000\n")
            f.write(f"map_Kd {texture_filename}\n")

        print(f"Model saved as {obj_filename} with texture {texture_filename}")

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
                            self.save_model()
                        elif event.key == pygame.K_r:
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
                    self.mesh_data = self.create_mesh(avg_points)

                # Clear screen
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # Draw camera feed
                glViewport(0, 0, self.display_width // 2, self.display_height)
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                glOrtho(-1, 1, -1, 1, -1, 1)
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                self.draw_camera_feed(frame)

                # Draw 3D model
                glViewport(self.display_width // 2, 0,
                          self.display_width // 2, self.display_height)
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(45, (self.display_width/2) / self.display_height, 0.1, 50.0)
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                glTranslatef(0.0, 0.0, -5)
                if self.mesh_data:
                    self.draw_mesh(self.mesh_data)

                pygame.display.flip()
                clock.tick(60)

        finally:
            if self.texture_id is not None:
                glDeleteTextures([self.texture_id])
            cap.release()
            pygame.quit()
            self.face_mesh.close()

if __name__ == "__main__":
    generator = FaceModelGenerator()
    generator.run()