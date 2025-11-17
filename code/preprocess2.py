import os
import cv2
import numpy as np

# Function to preprocess images and save them to output folder
def preprocess_and_save_images(input_folder, output_folder, target_size=(128, 128)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    preprocessed_images = []
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        if image is not None:
            # Resize image
            image = cv2.resize(image, target_size)
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0

            # Save preprocessed image to output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image * 255)  # Save as uint8 (0-255) after rescaling
            preprocessed_images.append(image)
    return preprocessed_images

# Preprocess and save moss images
preprocess_and_save_images(r"C:\Users\hp\Downloads\Internship_Research\Research\Research\omossimg1",
                           r"C:\Users\hp\Downloads\Internship_Research\Research\Research\omossimg2")

# Preprocess and save non-moss images
preprocess_and_save_images(r"C:\Users\hp\Downloads\Internship_Research\Research\Research\onmossimg1",
                           r"C:\Users\hp\Downloads\Internship_Research\Research\Research\onmossimg2")
