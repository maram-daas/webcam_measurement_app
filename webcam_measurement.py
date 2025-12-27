import cv2
import numpy as np
import pickle
import os
from pathlib import Path

class MeasurementApp:
    def __init__(self):
        self.cap = None
        self.points = []
        self.measurements = []
        self.mode = 'distance'
        self.calibrated = False
        self.focal_length = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.depth = 100
        self.reference_distance = None
        self.calibration_file = 'camera_calibration.pkl'
        self.current_frame = None
        self.is_image_mode = False
        self.loaded_image = None
        
        # Zoom/Pan features
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.original_frame = None
        
        # Selection for deletion
        self.selected_measurement = None
        self.hover_measurement = None
        
        # Input field state
        self.input_active = False
        self.input_text = ""
        self.input_prompt = ""
        self.pending_image_path = None
        
        # Modern color scheme
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.DARK_BG = (28, 28, 30)
        self.ACCENT_BLUE = (255, 170, 76)  # Warm gold-blue
        self.ACCENT_CYAN = (255, 200, 100)  # Bright cyan
        self.SUCCESS_GREEN = (100, 255, 150)
        self.WARNING_RED = (100, 120, 255)
        self.GLASS_GRAY = (60, 60, 65)
        self.GLASS_LIGHT = (90, 90, 95)
        
        # UI buttons
        self.buttons = []
        self.hover_button = None
        
        # UI state
        self.show_shortcuts = False
        
        # Load calibration if exists
        self.load_calibration()
    
    def load_calibration(self):
        """Load camera calibration from file"""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'rb') as f:
                    data = pickle.load(f)
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
                self.focal_length = data['focal_length']
                self.calibrated = True
                print("✓ Calibration loaded successfully")
            except Exception as e:
                print(f"⚠ Failed to load calibration: {e}")
    
    def save_calibration(self):
        """Save camera calibration to file"""
        try:
            data = {
                'camera_matrix': self.camera_matrix,
                'dist_coeffs': self.dist_coeffs,
                'focal_length': self.focal_length
            }
            with open(self.calibration_file, 'wb') as f:
                pickle.dump(data, f)
            print("✓ Calibration saved successfully")
        except Exception as e:
            print(f"⚠ Failed to save calibration: {e}")
    
    def simple_calibration(self, frame_width, frame_height):
        """Simple calibration using default values"""
        self.focal_length = frame_width * 1.2
        cx = frame_width / 2
        cy = frame_height / 2
        self.camera_matrix = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        self.calibrated = True
        self.save_calibration()
        print("✓ Simple calibration completed")
    
    def calibrate_with_reference(self, pixel_distance, real_distance_cm):
        """Calibrate using a known reference object"""
        self.reference_distance = real_distance_cm / pixel_distance
        print(f"✓ Reference calibration: {self.reference_distance:.4f} cm/pixel")
    
    def reset_calibration(self):
        """Reset to default calibration"""
        self.reference_distance = None
        print("✓ Reset to default calibration")
    
    def set_mode(self, mode):
        """Set measurement mode"""
        self.mode = mode
        self.points = []
        if mode == 'calibrate':
            self.measurements = []
        print(f"→ {mode.upper()} mode")
    
    def start_load_image(self):
        """Start image loading process"""
        self.input_prompt = "Enter image path:"
        self.input_text = ""
        self.input_active = True
        self.pending_image_path = True
    
    def switch_to_video(self):
        """Switch to video mode"""
        self.is_image_mode = False
        self.loaded_image = None
        self.points = []
        self.measurements = []
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        print("→ VIDEO mode")
    
    def reset_all(self):
        """Reset all points and measurements and depth"""
        self.points = []
        self.measurements = []
        self.depth = 100
        print("→ All measurements reset, depth reset to 100cm")
    
    def zoom_in(self):
        """Zoom in"""
        self.zoom_level = min(5.0, self.zoom_level + 0.2)
        print(f"→ Zoom: {self.zoom_level:.1f}x")
    
    def zoom_out(self):
        """Zoom out"""
        self.zoom_level = max(1.0, self.zoom_level - 0.2)
        if self.zoom_level == 1.0:
            self.pan_x = 0
            self.pan_y = 0
        print(f"→ Zoom: {self.zoom_level:.1f}x")
    
    def toggle_shortcuts(self):
        """Toggle keyboard shortcuts display"""
        self.show_shortcuts = not self.show_shortcuts
        print(f"→ Shortcuts {'shown' if self.show_shortcuts else 'hidden'}")
    
    def load_image(self, image_path):
        """Load an image for measurement"""
        try:
            image_path = image_path.strip().strip('"').strip("'")
            if not os.path.exists(image_path):
                print(f"⚠ File not found: {image_path}")
                return False
            
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                max_dimension = 1280
                if w > max_dimension or h > max_dimension:
                    scale = max_dimension / max(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img = cv2.resize(img, (new_w, new_h))
                    print(f"✓ Image resized to {new_w}x{new_h}")
                
                self.loaded_image = img
                self.is_image_mode = True
                self.points = []
                self.measurements = []
                self.zoom_level = 1.0
                self.pan_x = 0
                self.pan_y = 0
                print(f"✓ Image loaded: {image_path}")
                return True
            else:
                print(f"⚠ Failed to load image: {image_path}")
                return False
        except Exception as e:
            print(f"⚠ Error loading image: {e}")
            return False
    
    def screen_to_original_coords(self, x, y, frame_shape):
        """Convert screen coordinates to original frame coordinates"""
        h, w = frame_shape[:2]
        orig_x = int((x - w/2) / self.zoom_level + w/2 - self.pan_x / self.zoom_level)
        orig_y = int((y - h/2) / self.zoom_level + h/2 - self.pan_y / self.zoom_level)
        return (orig_x, orig_y)
    
    def original_to_screen_coords(self, x, y, frame_shape):
        """Convert original frame coordinates to screen coordinates"""
        h, w = frame_shape[:2]
        screen_x = int((x - w/2 + self.pan_x / self.zoom_level) * self.zoom_level + w/2)
        screen_y = int((y - h/2 + self.pan_y / self.zoom_level) * self.zoom_level + h/2)
        return (screen_x, screen_y)
    
    def apply_zoom_pan(self, frame):
        """Apply zoom and pan to frame"""
        if self.zoom_level == 1.0 and self.pan_x == 0 and self.pan_y == 0:
            return frame
        
        h, w = frame.shape[:2]
        M = np.float32([
            [self.zoom_level, 0, w/2 * (1 - self.zoom_level) + self.pan_x],
            [0, self.zoom_level, h/2 * (1 - self.zoom_level) + self.pan_y]
        ])
        zoomed = cv2.warpAffine(frame, M, (w, h))
        return zoomed
    
    def add_modern_vignette(self, frame):
        """Add subtle vignette effect"""
        h, w = frame.shape[:2]
        kernel_x = cv2.getGaussianKernel(w, w/3)
        kernel_y = cv2.getGaussianKernel(h, h/3)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        mask = np.stack([mask] * 3, axis=2)
        return (frame * (0.3 + 0.7 * mask)).astype(np.uint8)
    
    def draw_glassmorphic_panel(self, frame, x, y, w, h, alpha=0.3):
        """Draw modern glassmorphic panel"""
        overlay = frame[y:y+h, x:x+w].copy()
        overlay[:] = self.GLASS_GRAY
        cv2.addWeighted(overlay, alpha, frame[y:y+h, x:x+w], 1-alpha, 0, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), self.GLASS_LIGHT, 1)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if self.input_active:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in self.buttons:
                if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                    btn['action']()
                    return
        
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_button = None
            for i, btn in enumerate(self.buttons):
                if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                    self.hover_button = i
                    break
            
            if self.hover_button is None:
                orig_coords = self.screen_to_original_coords(x, y, self.original_frame.shape)
                self.hover_measurement = self.get_measurement_at_point(orig_coords)
            return
            
        orig_coords = self.screen_to_original_coords(x, y, self.original_frame.shape)
        
        if event == cv2.EVENT_RBUTTONDOWN:
            clicked_measurement = self.get_measurement_at_point(orig_coords)
            if clicked_measurement is not None:
                meas_type = self.measurements[clicked_measurement]['type']
                del self.measurements[clicked_measurement]
                self.selected_measurement = None
                print(f"✗ Deleted {meas_type} measurement")
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == 'calibrate':
                if len(self.points) < 2:
                    self.points.append(orig_coords)
                    print(f"Calibration point {len(self.points)}: {orig_coords}")
                    
                    if len(self.points) == 2:
                        pixel_dist = self.calculate_pixel_distance(self.points[0], self.points[1])
                        self.input_prompt = f"Pixel distance: {pixel_dist:.1f}px | Enter real distance in cm:"
                        self.input_active = True
                        self.input_text = ""
            
            elif self.mode == 'distance':
                self.points.append(orig_coords)
                print(f"Point {len(self.points)}: {orig_coords}")
                
                if len(self.points) == 2:
                    self.add_distance_measurement()
                    self.points = []
            
            elif self.mode == 'angle':
                self.points.append(orig_coords)
                print(f"Point {len(self.points)}: {orig_coords}")
                
                if len(self.points) == 3:
                    self.add_angle_measurement()
                    self.points = []
    
    def add_distance_measurement(self):
        """Add a distance measurement"""
        if len(self.points) >= 2:
            p1, p2 = self.points[0], self.points[1]
            pixel_dist = self.calculate_pixel_distance(p1, p2)
            real_dist = self.calculate_real_distance(pixel_dist)
            
            measurement = {
                'type': 'distance',
                'points': [p1, p2],
                'pixel_distance': pixel_dist,
                'real_distance': real_dist
            }
            self.measurements.append(measurement)
            print(f"✓ Distance measured: {real_dist:.2f} cm" if real_dist else f"✓ Distance: {pixel_dist:.1f} px")
    
    def add_angle_measurement(self):
        """Add an angle measurement"""
        if len(self.points) >= 3:
            p1, p2, p3 = self.points[0], self.points[1], self.points[2]
            angle = self.calculate_angle(p1, p2, p3)
            
            measurement = {
                'type': 'angle',
                'points': [p1, p2, p3],
                'angle': angle
            }
            self.measurements.append(measurement)
            print(f"✓ Angle measured: {angle:.1f}° (vertex at point 1)")
    
    def get_measurement_at_point(self, point):
        """Check if a point is near any measurement"""
        threshold = 20
        
        for i, meas in enumerate(self.measurements):
            if meas['type'] == 'distance':
                p1, p2 = meas['points']
                dist_to_line = self.point_to_line_distance(point, p1, p2)
                if dist_to_line < threshold:
                    return i
            
            elif meas['type'] == 'angle':
                vertex = meas['points'][0]
                dist_to_vertex = self.calculate_pixel_distance(point, vertex)
                if dist_to_vertex < threshold:
                    return i
        
        return None
    
    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        
        if denominator == 0:
            return float('inf')
        
        return numerator / denominator
    
    def remove_last_point(self):
        """Remove the last point"""
        if self.points:
            removed = self.points.pop()
            print(f"✗ Removed point: {removed}")
    
    def calculate_pixel_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def calculate_real_distance(self, pixel_distance):
        """Convert pixel distance to real-world distance in cm"""
        if self.reference_distance:
            return pixel_distance * self.reference_distance
        elif self.calibrated and self.focal_length:
            return (pixel_distance * self.depth) / self.focal_length
        else:
            return None
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle at vertex p1 between points p2 and p3"""
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p1[0], p3[1] - p1[1]])
        
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)
        
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0
        
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def find_optimal_text_position(self, p1_screen, p2_screen, frame_shape):
        """Find optimal position for distance label to avoid overlap"""
        h, w = frame_shape[:2]
        
        dx = p2_screen[0] - p1_screen[0]
        dy = p2_screen[1] - p1_screen[1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            perp_x = -dy / length
            perp_y = dx / length
        else:
            perp_x = 0
            perp_y = -1
        
        mid_x = (p1_screen[0] + p2_screen[0]) // 2
        mid_y = (p1_screen[1] + p2_screen[1]) // 2
        
        offset = 30
        pos_above = (int(mid_x + perp_x * offset), int(mid_y + perp_y * offset))
        pos_below = (int(mid_x - perp_x * offset), int(mid_y - perp_y * offset))
        
        if pos_above[1] > 60 and pos_above[1] < h - 180:
            return pos_above
        else:
            return pos_below
    
    def draw_text_with_background(self, frame, text, pos, font_scale=0.7, color=None, bg_color=None):
        """Draw text with semi-transparent background"""
        if color is None:
            color = self.ACCENT_CYAN
        if bg_color is None:
            bg_color = self.DARK_BG
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        padding = 8
        bg_x = pos[0] - padding
        bg_y = pos[1] - text_h - padding
        bg_w = text_w + 2 * padding
        bg_h = text_h + 2 * padding
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), bg_color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.rectangle(frame, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), color, 1)
        cv2.putText(frame, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)
    
    def draw_input_box(self, frame):
        """Draw modern input box"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), self.BLACK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        box_w = 700
        box_h = 180
        box_x = (w - box_w) // 2
        box_y = (h - box_h) // 2
        
        self.draw_glassmorphic_panel(frame, box_x, box_y, box_w, box_h, 0.4)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), self.ACCENT_CYAN, 2)
        
        cv2.putText(frame, self.input_prompt, (box_x + 20, box_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 2, cv2.LINE_AA)
        
        input_box_y = box_y + 75
        self.draw_glassmorphic_panel(frame, box_x + 20, input_box_y, box_w - 40, 50, 0.3)
        cv2.rectangle(frame, (box_x + 20, input_box_y), (box_x + box_w - 20, input_box_y + 50),
                     self.ACCENT_BLUE, 2)
        
        display_text = self.input_text + "│"
        cv2.putText(frame, display_text, (box_x + 35, input_box_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.WHITE, 2, cv2.LINE_AA)
        
        cv2.putText(frame, "ENTER to confirm | ESC to cancel", (box_x + 160, box_y + box_h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.GLASS_LIGHT, 1, cv2.LINE_AA)
    
    def draw_shortcuts_panel(self, frame):
        """Draw modern shortcuts panel"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), self.BLACK, -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        
        panel_w = 700
        panel_h = 520
        panel_x = (w - panel_w) // 2
        panel_y = (h - panel_h) // 2
        
        self.draw_glassmorphic_panel(frame, panel_x, panel_y, panel_w, panel_h, 0.4)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), self.ACCENT_CYAN, 3)
        
        cv2.putText(frame, "KEYBOARD SHORTCUTS", (panel_x + 180, panel_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.ACCENT_CYAN, 2, cv2.LINE_AA)
        
        cv2.line(frame, (panel_x + 20, panel_y + 60), (panel_x + panel_w - 20, panel_y + 60), 
                self.ACCENT_BLUE, 2)
        
        shortcuts = [
            ("Q", "Exit application"),
            ("D", "Distance measurement mode"),
            ("A", "Angle measurement mode"),
            ("C", "Calibration mode"),
            ("X", "Reset calibration"),
            ("L", "Load image file"),
            ("V", "Switch to live video"),
            ("R", "Clear all measurements"),
            ("BACKSPACE", "Remove last point"),
            ("I / O", "Adjust depth (+/-10cm)"),
            ("+ / -", "Zoom in/out"),
            ("1/2/3/4", "Pan: Left/Right/Up/Down"),
            ("H", "Toggle shortcuts panel"),
            ("RIGHT CLICK", "Delete measurement"),
        ]
        
        y_pos = panel_y + 95
        for key, desc in shortcuts:
            self.draw_glassmorphic_panel(frame, panel_x + 30, y_pos - 20, 150, 25, 0.3)
            cv2.rectangle(frame, (panel_x + 30, y_pos - 20), (panel_x + 180, y_pos + 5), 
                         self.ACCENT_BLUE, 2)
            cv2.putText(frame, key, (panel_x + 45, y_pos - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ACCENT_CYAN, 1, cv2.LINE_AA)
            
            cv2.putText(frame, desc, (panel_x + 200, y_pos - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1, cv2.LINE_AA)
            
            y_pos += 28
        
        cv2.putText(frame, "Press H to close", (panel_x + 270, panel_y + panel_h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.GLASS_LIGHT, 1, cv2.LINE_AA)
    
    def draw_path_input_box(self, frame):
        """Draw modern path input box"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), self.BLACK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        box_w = 800
        box_h = 180
        box_x = (w - box_w) // 2
        box_y = (h - box_h) // 2
        
        self.draw_glassmorphic_panel(frame, box_x, box_y, box_w, box_h, 0.4)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), self.ACCENT_CYAN, 2)
        
        cv2.putText(frame, "LOAD IMAGE", (box_x + 20, box_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ACCENT_CYAN, 2, cv2.LINE_AA)
        
        input_box_y = box_y + 65
        self.draw_glassmorphic_panel(frame, box_x + 20, input_box_y, box_w - 40, 50, 0.3)
        cv2.rectangle(frame, (box_x + 20, input_box_y), (box_x + box_w - 20, input_box_y + 50),
                     self.ACCENT_BLUE, 2)
        
        display_text = self.input_text[-60:] if len(self.input_text) > 60 else self.input_text
        display_text += "│"
        cv2.putText(frame, display_text, (box_x + 30, input_box_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 1, cv2.LINE_AA)
        
        cv2.putText(frame, "ENTER to load | ESC to cancel", (box_x + 240, box_y + box_h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.GLASS_LIGHT, 1, cv2.LINE_AA)
    
    def draw_ui(self, frame):
        """Draw modern UI elements"""
        h, w = frame.shape[:2]
        
        # Top panel with glassmorphism
        panel_height = 55
        self.draw_glassmorphic_panel(frame, 0, 0, w, panel_height, 0.4)
        
        # Live indicator
        if not self.is_image_mode:
            cv2.circle(frame, (w - 35, 28), 6, self.WARNING_RED, -1)
            cv2.circle(frame, (w - 35, 28), 9, (150, 180, 255), 2)
            cv2.putText(frame, "LIVE", (w - 80, 33),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1, cv2.LINE_AA)
        
        # Mode display
        mode_display = "IMAGE" if self.is_image_mode else "LIVE"
        mode_map = {
            'calibrate': 'CALIBRATION',
            'distance': 'DISTANCE',
            'angle': 'ANGLE'
        }
        mode_text = f"{mode_display} | {mode_map[self.mode]}"
        
        cv2.putText(frame, mode_text, (25, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.ACCENT_CYAN, 2, cv2.LINE_AA)
        
        # Calibration badge
        if self.reference_distance:
            calib_text = "[CUSTOM]"
            calib_color = self.SUCCESS_GREEN
        elif self.calibrated:
            calib_text = "[DEFAULT]"
            calib_color = self.ACCENT_BLUE
        else:
            calib_text = "[UNCALIBRATED]"
            calib_color = self.WARNING_RED
        
        (text_w, text_h), _ = cv2.getTextSize(calib_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Move further left to not hide LIVE indicator
        calib_x = w - text_w - 120 if not self.is_image_mode else w - text_w - 20
        cv2.putText(frame, calib_text, (calib_x, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, calib_color, 2, cv2.LINE_AA)
        
        # Bottom modern panel
        bottom_height = 145
        self.draw_glassmorphic_panel(frame, 0, h - bottom_height, w, bottom_height, 0.4)
        
        # Instructions
        info_y = h - bottom_height + 25
        mode_instructions = {
            'calibrate': "Click 2 points on object of known size",
            'distance': "Click 2 points to measure distance",
            'angle': "Click 3 points: VERTEX FIRST, then two arms"
        }
        info_text = mode_instructions.get(self.mode, "")
        
        cv2.putText(frame, info_text, (25, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.WHITE, 2, cv2.LINE_AA)
        
        # Separator
        cv2.line(frame, (25, h - 110), (w - 25, h - 110), self.ACCENT_BLUE, 1)
        
        # Create modern buttons
        self.buttons = []
        button_y = h - 95
        button_height = 32
        button_spacing = 8
        
        buttons_row1 = [
            {'label': 'Distance', 'action': lambda: self.set_mode('distance')},
            {'label': 'Angle', 'action': lambda: self.set_mode('angle')},
            {'label': 'Calibrate', 'action': lambda: self.set_mode('calibrate')},
            {'label': 'Reset Cal', 'action': self.reset_calibration},
            {'label': 'Load Img', 'action': self.start_load_image},
            {'label': 'Video', 'action': self.switch_to_video},
        ]
        
        buttons_row2 = [
            {'label': 'Remove Pt', 'action': self.remove_last_point},
            {'label': 'Reset All', 'action': self.reset_all},
            {'label': 'Zoom +', 'action': self.zoom_in},
            {'label': 'Zoom -', 'action': self.zoom_out},
            {'label': 'Help (H)', 'action': self.toggle_shortcuts},
        ]
        
        # Draw row 1
        x_pos = 25
        for btn_info in buttons_row1:
            (text_w, text_h), _ = cv2.getTextSize(btn_info['label'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            btn_w = text_w + 24
            
            btn = {
                'x': x_pos,
                'y': button_y,
                'w': btn_w,
                'h': button_height,
                'label': btn_info['label'],
                'action': btn_info['action']
            }
            self.buttons.append(btn)
            
            is_hovered = (self.hover_button == len(self.buttons) - 1)
            
            if is_hovered:
                self.draw_glassmorphic_panel(frame, x_pos, button_y, btn_w, button_height, 0.5)
                cv2.rectangle(frame, (x_pos, button_y), (x_pos + btn_w, button_y + button_height),
                            self.ACCENT_CYAN, 2)
            else:
                self.draw_glassmorphic_panel(frame, x_pos, button_y, btn_w, button_height, 0.3)
                cv2.rectangle(frame, (x_pos, button_y), (x_pos + btn_w, button_y + button_height),
                            self.GLASS_LIGHT, 1)
            
            text_x = x_pos + 12
            text_y = button_y + 21
            text_color = self.ACCENT_CYAN if is_hovered else self.WHITE
            cv2.putText(frame, btn_info['label'], (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            
            x_pos += btn_w + button_spacing
        
        # Draw row 2
        button_y2 = button_y + button_height + 6
        x_pos = 25
        for btn_info in buttons_row2:
            (text_w, text_h), _ = cv2.getTextSize(btn_info['label'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            btn_w = text_w + 24
            
            btn = {
                'x': x_pos,
                'y': button_y2,
                'w': btn_w,
                'h': button_height,
                'label': btn_info['label'],
                'action': btn_info['action']
            }
            self.buttons.append(btn)
            
            is_hovered = (self.hover_button == len(self.buttons) - 1)
            
            if is_hovered:
                self.draw_glassmorphic_panel(frame, x_pos, button_y2, btn_w, button_height, 0.5)
                cv2.rectangle(frame, (x_pos, button_y2), (x_pos + btn_w, button_y2 + button_height),
                            self.ACCENT_CYAN, 2)
            else:
                self.draw_glassmorphic_panel(frame, x_pos, button_y2, btn_w, button_height, 0.3)
                cv2.rectangle(frame, (x_pos, button_y2), (x_pos + btn_w, button_y2 + button_height),
                            self.GLASS_LIGHT, 1)
            
            text_x = x_pos + 12
            text_y = button_y2 + 21
            text_color = self.ACCENT_CYAN if is_hovered else self.WHITE
            cv2.putText(frame, btn_info['label'], (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            
            x_pos += btn_w + button_spacing
        
        # Status info
        status_x = w - 320
        status_text = f"Pts: {len(self.points)} | Meas: {len(self.measurements)} | Zoom: {self.zoom_level:.1f}x"
        cv2.putText(frame, status_text, (status_x, button_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1, cv2.LINE_AA)
        
        if not self.reference_distance:
            depth_text = f"Depth: {self.depth}cm (I/O)"
            cv2.putText(frame, depth_text, (status_x, button_y2 + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1, cv2.LINE_AA)
    
    def draw_measurements(self, frame):
        """Draw all measurements with modern styling"""
        h, w = frame.shape[:2]
        
        for i, meas in enumerate(self.measurements):
            is_hovered = (i == self.hover_measurement)
            
            if meas['type'] == 'distance':
                color = self.ACCENT_CYAN if is_hovered else self.WHITE
                thickness = 3 if is_hovered else 2
                
                p1_screen = self.original_to_screen_coords(meas['points'][0][0], meas['points'][0][1], self.original_frame.shape)
                p2_screen = self.original_to_screen_coords(meas['points'][1][0], meas['points'][1][1], self.original_frame.shape)
                
                cv2.line(frame, p1_screen, p2_screen, color, thickness, cv2.LINE_AA)
                
                # Draw endpoints
                cv2.circle(frame, p1_screen, 5, color, -1)
                cv2.circle(frame, p2_screen, 5, color, -1)
                
                # Position text
                text_pos = self.find_optimal_text_position(p1_screen, p2_screen, frame.shape)
                
                if meas['real_distance']:
                    text = f"{meas['real_distance']:.2f} cm"
                else:
                    text = f"{meas['pixel_distance']:.1f} px"
                
                self.draw_text_with_background(frame, text, text_pos, font_scale=0.65, color=color)
            
            elif meas['type'] == 'angle':
                color = self.ACCENT_CYAN if is_hovered else self.WARNING_RED
                thickness = 3 if is_hovered else 2
                
                vertex_screen = self.original_to_screen_coords(meas['points'][0][0], meas['points'][0][1], self.original_frame.shape)
                p2_screen = self.original_to_screen_coords(meas['points'][1][0], meas['points'][1][1], self.original_frame.shape)
                p3_screen = self.original_to_screen_coords(meas['points'][2][0], meas['points'][2][1], self.original_frame.shape)
                
                cv2.line(frame, vertex_screen, p2_screen, color, thickness, cv2.LINE_AA)
                cv2.line(frame, vertex_screen, p3_screen, color, thickness, cv2.LINE_AA)
                
                # Draw vertex
                cv2.circle(frame, vertex_screen, 6, color, -1)
                
                # Calculate angle for arc
                v1 = np.array([p2_screen[0] - vertex_screen[0], p2_screen[1] - vertex_screen[1]])
                v2 = np.array([p3_screen[0] - vertex_screen[0], p3_screen[1] - vertex_screen[1]])
                
                angle1 = np.degrees(np.arctan2(v1[1], v1[0]))
                angle2 = np.degrees(np.arctan2(v2[1], v2[0]))
                
                if angle1 < 0:
                    angle1 += 360
                if angle2 < 0:
                    angle2 += 360
                
                start_angle = min(angle1, angle2)
                end_angle = max(angle1, angle2)
                
                if end_angle - start_angle > 180:
                    start_angle, end_angle = end_angle, start_angle + 360
                
                radius = int(40 * self.zoom_level)
                cv2.ellipse(frame, vertex_screen, (radius, radius), 0, start_angle, end_angle, color, thickness, cv2.LINE_AA)
                
                # Draw angle text
                text = f"{meas['angle']:.1f}°"
                text_x = vertex_screen[0] + int(55 * self.zoom_level)
                text_y = vertex_screen[1] - int(15 * self.zoom_level)
                
                self.draw_text_with_background(frame, text, (text_x, text_y), font_scale=0.65, color=color)
        
        # Draw current points
        for i, pt in enumerate(self.points):
            pt_screen = self.original_to_screen_coords(pt[0], pt[1], self.original_frame.shape)
            radius = int(6 * self.zoom_level)
            cv2.circle(frame, pt_screen, radius, self.ACCENT_CYAN, -1)
            cv2.circle(frame, pt_screen, int(9 * self.zoom_level), self.ACCENT_CYAN, 2, cv2.LINE_AA)
            
            label = str(i + 1)
            if self.mode == 'angle' and i == 0:
                label += " (V)"
            
            label_pos = (pt_screen[0] + 15, pt_screen[1] - 15)
            self.draw_text_with_background(frame, label, label_pos, font_scale=0.5, color=self.ACCENT_CYAN)
            
            if i > 0:
                prev_pt_screen = self.original_to_screen_coords(self.points[i-1][0], self.points[i-1][1], self.original_frame.shape)
                cv2.line(frame, prev_pt_screen, pt_screen, self.ACCENT_CYAN, 2, cv2.LINE_AA)
    
    def handle_input_key(self, key):
        """Handle keyboard input for text entry"""
        if key == 13:  # Enter
            if self.pending_image_path is not None:
                if self.load_image(self.input_text):
                    self.input_active = False
                    self.pending_image_path = None
                    self.mode = 'distance'
                    self.points = []
                else:
                    self.input_text = ""
            else:  # Calibration
                try:
                    real_dist = float(self.input_text)
                    if real_dist > 0:
                        pixel_dist = self.calculate_pixel_distance(self.points[0], self.points[1])
                        self.calibrate_with_reference(pixel_dist, real_dist)
                        print(f"✓ Calibration successful: {real_dist} cm")
                        self.input_active = False
                        self.mode = 'distance'
                        self.points = []
                    else:
                        print("⚠ Distance must be positive")
                except ValueError:
                    print("⚠ Invalid number")
        
        elif key == 27:  # Escape
            self.input_active = False
            self.pending_image_path = None
            self.mode = 'distance'
            self.points = []
            print("✗ Action cancelled")
        
        elif key == 8:  # Backspace
            self.input_text = self.input_text[:-1]
        
        elif 32 <= key <= 126:  # Printable characters
            self.input_text += chr(key)
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*70)
        print(" ✨ ADVANCED COMPUTER VISION MEASUREMENT SYSTEM")
        print("="*70)
        print("\n📌 FEATURES:")
        print(" • Modern glassmorphic UI with smooth interactions")
        print(" • Click to add measurement points (2 for distance, 3 for angle)")
        print(" • ANGLE MODE: First point = VERTEX, then two arm points")
        print(" • Right-click measurements to delete them")
        print(" • Load images or use live webcam")
        print(" • Zoom (+/-) and Pan (1=Left 2=Right 3=Up 4=Down)")
        print(" • Press H for full keyboard shortcuts")
        print("\n" + "="*70 + "\n")
        
        # Open webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        ret, frame = self.cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"📷 Camera resolution: {w}x{h}")
            if not self.calibrated:
                print("⚙️  Performing default calibration...")
                self.simple_calibration(w, h)
        
        cv2.namedWindow('Measurement App', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Measurement App', self.mouse_callback)
        
        print("\n✓ App is running! Press Q to quit\n")
        
        while True:
            # Get frame
            if self.is_image_mode and self.loaded_image is not None:
                frame = self.loaded_image.copy()
            else:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Cannot read frame")
                    break
                frame = cv2.flip(frame, 1)
            
            # Store original frame
            self.original_frame = frame.copy()
            
            # Apply zoom and pan
            frame = self.apply_zoom_pan(frame)
            
            # Draw measurements and UI
            self.draw_measurements(frame)
            self.draw_ui(frame)
            
            # Draw overlays
            if self.show_shortcuts:
                self.draw_shortcuts_panel(frame)
            
            if self.input_active:
                if self.pending_image_path is not None:
                    self.draw_path_input_box(frame)
                else:
                    self.draw_input_box(frame)
            
            cv2.imshow('Measurement App', frame)
            
            # Check if window is closed
            if cv2.getWindowProperty('Measurement App', cv2.WND_PROP_VISIBLE) < 1:
                break
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if self.input_active:
                self.handle_input_key(key)
                continue
            
            # Normal shortcuts
            if key == ord('q'):
                break
            elif key == ord('h'):
                self.toggle_shortcuts()
            elif key == ord('d'):
                self.set_mode('distance')
            elif key == ord('a'):
                self.set_mode('angle')
            elif key == ord('c'):
                self.set_mode('calibrate')
            elif key == ord('x'):
                self.reset_calibration()
            elif key == ord('l'):
                self.start_load_image()
            elif key == ord('v'):
                self.switch_to_video()
            elif key == ord('r'):
                self.reset_all()
            elif key == 8:  # Backspace
                self.remove_last_point()
            elif key == ord('i') and not self.reference_distance:
                self.depth += 10
                print(f"→ Depth: {self.depth}cm")
            elif key == ord('o') and not self.reference_distance:
                self.depth = max(10, self.depth - 10)
                print(f"→ Depth: {self.depth}cm")
            elif key == ord('+') or key == ord('='):
                self.zoom_in()
            elif key == ord('-') or key == ord('_'):
                self.zoom_out()
            elif key == ord('1'):  # Pan left
                if self.zoom_level > 1.0:
                    self.pan_x += 20
                    print("→ Pan left")
            elif key == ord('2'):  # Pan right
                if self.zoom_level > 1.0:
                    self.pan_x -= 20
                    print("→ Pan right")
            elif key == ord('3'):  # Pan up
                if self.zoom_level > 1.0:
                    self.pan_y += 20
                    print("→ Pan up")
            elif key == ord('4'):  # Pan down
                if self.zoom_level > 1.0:
                    self.pan_y -= 20
                    print("→ Pan down")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ App closed successfully.\n")

if __name__ == "__main__":
    app = MeasurementApp()
    app.run()
