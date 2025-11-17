import cv2
import numpy as np
from playsound import playsound
import tensorflow as tf

# Load the trained ML model
ml_model = tf.keras.models.load_model(r"C:\Users\hp\Downloads\Internship_Research\Research\Research\code\moss_detection_model.keras")

def detect_moss_ml(image):
    # Preprocess the input image for the ML model
    processed_image = cv2.resize(image, (128, 128)) / 255.0
    
    # Perform inference with the ML model
    prediction = ml_model.predict(np.expand_dims(processed_image, axis=0))
    
    # If prediction is close to 1, consider it as moss
    if prediction[0][0] > 0.6:
        return True, (0, 0, image.shape[1], image.shape[0])  # Just return the full image area
    return False, None

def detect_moss(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color range for moss (adjusted values)
    lower_green = np.array([25, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    # Threshold the image to get only moss pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contours are found
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Calculate contour perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Check for circularity and area to filter out false positives
        if circularity < 0.6 and area > 1000:
            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            
            # Check for solidity to further filter out false positives
            if solidity > 0.6:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                return True, (x, y, x+w, y+h)
    
    return False, None

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use the default camera (usually webcam)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return
    
    # Delay between frame captures (in milliseconds)
    delay = 250
    
    while True:
        # Read frame from the camera
        ret, frame = cap.read()
        
        # Check if the frame is successfully captured
        if not ret:
            print("Error: Unable to capture frame")
            break
        
        # Check for moss using ML model
        moss_detected, bbox_ml = detect_moss_ml(frame)
        moss_detected_color, bbox_color = detect_moss(frame)
        
        if moss_detected or moss_detected_color:
            print("Moss detected! Alerting user...")
            # Play audio alert
            # playsound('alert_sound.mp3')  # Replace 'alert_sound.mp3' with your audio file
            
            # Draw red box and label
            if moss_detected:
                x1, y1, x2, y2 = bbox_ml
            elif moss_detected_color:
                x1, y1, x2, y2 = bbox_color
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, 'Moss', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            print("No moss detected.")
        
        # Display the frame
        cv2.imshow('Moss Detection', frame)
        
        # Wait for the specified delay before reading the next frame
        key = cv2.waitKey(delay)
        
        # Check for key press to exit
        if key & 0xFF == ord('q'):
            break
    
    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
