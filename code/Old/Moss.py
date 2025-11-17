import cv2
import numpy as np
from playsound import playsound

def detect_moss(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color range for green (you may need to adjust these values)
    lower_green = np.array([20, 50, 50])
    upper_green = np.array([100, 255, 255])
    
    # Threshold the image to get only green pixels
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Count green pixels
    green_pixel_count = np.sum(mask_green) // 255
    
    # Calculate proportion of green pixels in the image
    total_pixels = image.shape[0] * image.shape[1]
    green_proportion = green_pixel_count / total_pixels
    
    # If green proportion is below a certain threshold, return False (no moss)
    if green_proportion < 0.05:  # Adjust threshold as needed
        return False
    
    # Define color range for moss (you may need to adjust these values)
    lower_moss = np.array([20, 50, 50])
    upper_moss = np.array([100, 255, 255])
    
    # Threshold the image to get only moss pixels
    mask_moss = cv2.inRange(hsv, lower_moss, upper_moss)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask_moss = cv2.morphologyEx(mask_moss, cv2.MORPH_OPEN, kernel)
    mask_moss = cv2.morphologyEx(mask_moss, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask_moss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contours are found
    moss_detected = False
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Calculate contour perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Check for circularity and area to filter out false positives
        if circularity < 0.7 and area > 500:
            moss_detected = True
            break
    
    return moss_detected

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use the default camera (usually webcam)
    
    while True:
        # Read frame from the camera
        ret, frame = cap.read()
        
        # Check if the frame is successfully captured
        if not ret:
            print("Error: Unable to capture frame")
            break
        
        # Check for moss in the frame
        if detect_moss(frame):
            print("Moss detected! Alerting user...")
            # Play audio alert
            playsound('C:\Brian\Research\moss_man.mp3')  # Replace 'alert_sound.mp3' with your audio file
            
        # Display the frame
        cv2.imshow('Moss Detection', frame)
        
        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
