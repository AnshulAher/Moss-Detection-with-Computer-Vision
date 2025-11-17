import os
import numpy as np
from PIL import Image

# Define a function to preprocess images
def preprocess_images(input_folder, output_folder, target_size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        # Load image
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert image to numpy array and normalize pixel values
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        
        # Save preprocessed image
        output_path = os.path.join(output_folder, filename)
        Image.fromarray((image * 255).astype(np.uint8)).save(output_path)

# Replace placeholders with actual values
preprocess_images(r"C:\Users\hp\Downloads\Internship_Research\Research\Research\nmoss_images", r"C:\Users\hp\Downloads\Internship_Research\Research\Research\onmossimg1")
