import cv2
import numpy as np
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import time
from scipy.spatial import Delaunay
import mss
from PIL import Image, ImageTk
import threading


class FaceModelGenerator:
    def __init__(self):
        # Initialize screen capture with MacBook Air M2 resolution
        self.capture_area = {
            'left': 0,
            'top': 0,
            'width': 2560,  # MacBook Air M2 width
            'height': 1600  # MacBook Air M2 height
        }

        # Initialize screen capture
        self.sct = mss.mss()

        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )

        # Initialize display
        pygame.init()
        self.display_width = 1400
        self.display_height = 800
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        self.window_surface = pygame.display.set_mode(
            (self.display_width, self.display_height),
            DOUBLEBUF | OPENGL | RESIZABLE
        )
        pygame.display.set_caption("Face Model Generator")

        # OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 1, 0])
        glEnable(GL_COLOR_MATERIAL)

        # Initialize tracking
        self.points_buffer = []
        self.max_buffer_size = 15
        self.mesh_data = None
        self.rotation = [0, 0, 0]
        self.running = True
        self.texture_id = None
        self.last_frame = None
        self.recording = False
        self.countdown_active = False

        # Performance optimization
        self.target_fps = 30
        self.frame_count = 0
        self.process_resolution = (2560, 1600)  # Full resolution processing
        self.display_resolution = (2560, 1600)
        self.camera_surface = pygame.Surface((2560, 1600))

        # Countdown font
        pygame.font.init()
        self.font = pygame.font.Font(None, 74)

    def start_countdown(self):
        """Start 5-second countdown before recording"""
        def countdown():
            for i in range(5, 0, -1):
                if not self.running:
                    return
                self.countdown_text = str(i)
                time.sleep(1)
            self.countdown_active = False
            self.recording = True
            self.countdown_text = ""

        self.countdown_active = True
        self.countdown_text = "5"
        threading.Thread(target=countdown, daemon=True).start()

    def draw_countdown(self):
        """Draw countdown text on screen"""
        if self.countdown_active:
            text_surface = self.font.render(
                self.countdown_text, True, (255, 0, 0))
            text_data = pygame.image.tostring(text_surface, 'RGB', 1)

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Save current matrix
            glPushMatrix()
            glLoadIdentity()

            # Draw text
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, text_surface.get_width(), text_surface.get_height(),
                         0, GL_RGB, GL_UNSIGNED_BYTE, text_data)

            glBegin(GL_QUADS)
            glTexCoord2f(0, 0)
            glVertex2f(-0.2, 0.2)
            glTexCoord2f(1, 0)
            glVertex2f(0.2, 0.2)
            glTexCoord2f(1, 1)
            glVertex2f(0.2, -0.2)
            glTexCoord2f(0, 1)
            glVertex2f(-0.2, -0.2)
            glEnd()

            # Restore matrix
            glPopMatrix()
            glDeleteTextures([texture])
            glDisable(GL_BLEND)

    def run(self):
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
                        elif event.key == pygame.K_SPACE and not self.recording and not self.countdown_active:
                            self.start_countdown()
                    elif event.type == pygame.MOUSEMOTION:
                        if event.buttons[0]:  # Left mouse button
                            self.rotation[1] += event.rel[0]
                            self.rotation[0] += event.rel[1]

                # Process screen capture if recording
                if self.recording:
                    frame = self.capture_screen()

                    # Process frame
                    points = self.process_frame(frame)
                    if points is not None:
                        self.points_buffer.append(points)
                        if len(self.points_buffer) > self.max_buffer_size:
                            self.points_buffer.pop(0)
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

                if self.recording:
                    self.draw_camera_feed(frame)

                # Draw 3D model
                glViewport(self.display_width // 2, 0,
                           self.display_width // 2, self.display_height)
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(45, (self.display_width/2) /
                               self.display_height, 0.1, 50.0)
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                glTranslatef(0.0, 0.0, -5)
                if self.mesh_data:
                    self.draw_mesh(self.mesh_data)

                # Draw countdown if active
                if self.countdown_active:
                    self.draw_countdown()

                pygame.display.flip()
                clock.tick(self.target_fps)

        finally:
            if self.texture_id is not None:
                glDeleteTextures([self.texture_id])
            self.sct.close()
            pygame.quit()
            self.face_mesh.close()

    def capture_screen(self):
        """Capture the selected screen area"""
        screenshot = self.sct.grab(self.capture_area)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def resize_frame(self, frame, target_size):
        """Resize frame to exact dimensions"""
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    def draw_camera_feed(self, frame):
        """Draw the camera feed on the left side"""
        if frame is None:
            return

        # Resize frame to match surface dimensions exactly
        frame_resized = cv2.resize(frame,
                                   (self.camera_surface.get_width(),
                                    self.camera_surface.get_height()),
                                   interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Create temporary surface if dimensions don't match
        temp_surface = pygame.surfarray.make_surface(frame_rgb)
        self.camera_surface.blit(temp_surface, (0, 0))

        texture_data = pygame.image.tostring(self.camera_surface, 'RGB', 1)

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     self.camera_surface.get_width(),
                     self.camera_surface.get_height(),
                     0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)

        glEnable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING)
        glColor3f(1, 1, 1)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 1)
        glVertex2f(-1, -1)
        glTexCoord2f(1, 1)
        glVertex2f(1, -1)
        glTexCoord2f(1, 0)
        glVertex2f(1, 1)
        glTexCoord2f(0, 0)
        glVertex2f(-1, 1)
        glEnd()

        glEnable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([texture_id])

    def process_frame(self, frame):
        """Process webcam frame and extract face landmarks"""
        # Resize frame for processing
        proc_frame = self.resize_frame(frame, self.process_resolution)
        frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

        # Store resized frame for texture
        self.last_frame = frame_rgb

        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        points = []
        for landmark in results.multi_face_landmarks[0].landmark:
            points.append([
                landmark.x - 0.5,
                0.5 - landmark.y,  # Flip Y coordinate
                -landmark.z        # Invert Z to face camera
            ])

        return np.array(points)

    def create_mesh(self, points):
        """Create triangulated mesh from points"""
        points_2d = points[:, :2]
        tri = Delaunay(points_2d)

        texture_coords = np.zeros_like(points_2d)
        texture_coords[:, 0] = points_2d[:, 0] + 0.5
        texture_coords[:, 1] = points_2d[:, 1] + 0.5

        normals = self.calculate_normals(points, tri.simplices)

        return {
            'vertices': points,
            'triangles': tri.simplices,
            'normals': normals,
            'texture_coords': texture_coords
        }

    def calculate_normals(self, vertices, triangles):
        """Calculate surface normals for lighting"""
        normals = np.zeros_like(vertices)

        for triangle in triangles:
            v1 = vertices[triangle[1]] - vertices[triangle[0]]
            v2 = vertices[triangle[2]] - vertices[triangle[0]]
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-10)

            normals[triangle[0]] += normal
            normals[triangle[1]] += normal
            normals[triangle[2]] += normal

        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1
        normals = normals / norms[:, np.newaxis]

        return normals

    def update_texture(self, frame):
        """Update OpenGL texture with current frame"""
        if frame is None:
            return

        frame_surface = pygame.surfarray.make_surface(frame)
        texture_data = pygame.image.tostring(frame_surface, 'RGB', 1)

        if self.texture_id is None:
            self.texture_id = glGenTextures(1)

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

        # Apply rotation
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)

        # Set initial orientation
        glRotatef(180, 0, 1, 0)  # Rotate to face camera

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

    def save_model(self):
        """Save the current model to OBJ file with texture"""
        if not self.mesh_data or self.last_frame is None:
            return

        timestamp = int(time.time())
        obj_filename = f"face_model_{timestamp}.obj"
        texture_filename = f"face_texture_{timestamp}.png"

        # Save texture
        cv2.imwrite(texture_filename, cv2.cvtColor(
            self.last_frame, cv2.COLOR_RGB2BGR))

        # Write OBJ file
        with open(obj_filename, 'w') as f:
            f.write(f"mtllib {obj_filename}.mtl\n")

            for vertex in self.mesh_data['vertices']:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            for texcoord in self.mesh_data['texture_coords']:
                f.write(f"vt {texcoord[0]} {texcoord[1]}\n")

            for normal in self.mesh_data['normals']:
                f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

            f.write("usemtl face_material\n")
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


if __name__ == "__main__":
    generator = FaceModelGenerator()
    generator.run()
