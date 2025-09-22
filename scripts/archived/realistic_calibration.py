import cv2
import numpy as np

class CalibratedAnalyzer:
    """Realistic approach with field calibration."""
    
    def __init__(self):
        # Known field dimensions (FIFA standard)
        self.field_length = 105  # meters
        self.field_width = 68   # meters
        
        # Field landmarks for calibration (in meters)
        self.real_points = np.array([
            [0, 0],           # Corner
            [0, 68],          # Corner
            [105, 68],        # Corner
            [105, 0],         # Corner
            [52.5, 34],       # Center
            [16.5, 34],       # Penalty area
            [88.5, 34],       # Penalty area
        ], dtype=np.float32)
        
    def calibrate_from_frame(self, frame):
        """Manual calibration - click on known points."""
        print("Pro kalibraci klikni na:")
        print("1. Levý horní roh")
        print("2. Levý dolní roh")
        print("3. Pravý dolní roh")
        print("4. Pravý horní roh")
        
        clicked_points = []
        
        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_points.append([x, y])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                cv2.imshow('Calibration', frame)
        
        cv2.imshow('Calibration', frame)
        cv2.setMouseCallback('Calibration', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if len(clicked_points) >= 4:
            # Calculate homography
            src = np.array(clicked_points[:4], dtype=np.float32)
            dst = self.real_points[:4] * 10  # Scale for visualization
            
            self.homography_matrix, _ = cv2.findHomography(src, dst)
            print("Kalibrace dokončena!")
            return True
        return False
    
    def pixel_to_real_distance(self, point1, point2):
        """Convert pixel distance to real meters using homography."""
        if not hasattr(self, 'homography_matrix'):
            return None
            
        # Transform points to real world coordinates
        p1_real = cv2.perspectiveTransform(
            np.array([[point1]], dtype=np.float32), 
            self.homography_matrix
        )[0][0]
        
        p2_real = cv2.perspectiveTransform(
            np.array([[point2]], dtype=np.float32),
            self.homography_matrix
        )[0][0]
        
        # Calculate real distance
        distance = np.sqrt((p2_real[0]-p1_real[0])**2 + 
                          (p2_real[1]-p1_real[1])**2)
        
        return distance / 10  # Unscale

# Ukázka správné kalibrace
print("Veo používá:")
print("1. Známé rozměry hřiště (105x68m)")
print("2. Detekce čar pomocí Hough transform")
print("3. Homography matrix pro perspektivní transformaci")
print("4. Každý pixel má různou hodnotu podle pozice")
print("\nReálné hodnoty pro fotbalistu:")
print("- Průměrná vzdálenost: 9-11 km za zápas (90 min)")
print("- Průměrná rychlost: 7-8 km/h")
print("- Sprint: 25-30 km/h")
print("- Za 10 sekund: ~20-30 metrů")
