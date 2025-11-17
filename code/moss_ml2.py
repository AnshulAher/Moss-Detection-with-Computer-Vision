_import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, models

# Function to load, resize, and convert images to RGB
def load_images_from_folder(folder, label, target_size=(128, 128)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = cv2.imread(img_path)
        if image is not None:
            # Resize image to target size
            image = cv2.resize(image, target_size)
            # Convert to RGB (for 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            labels.append(label)
    return images, labels

# Load moss and non-moss images
moss_images, moss_labels = load_images_from_folder(r"Give_omossimg2_folder_path", label=1)
non_moss_images, non_moss_labels = load_images_from_folder(r"Give_onmossimg2_folder_path", label=0)

# Combine images and labels
images = np.array(moss_images + non_moss_images)
labels = np.array(moss_labels + non_moss_labels)

# Shuffle dataset
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

# Split dataset into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data Augmentation (using OpenCV)
def augment_image(image):
    # Random horizontal flip
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    # Random rotation (between -15 and 15 degrees)
    angle = np.random.randint(-15, 15)
    rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return image

# Augment training images
augmented_train_images = [augment_image(image) for image in train_images]

# Convert to numpy array
augmented_train_images = np.array(augmented_train_images)

# Preprocessing (normalize pixel values)
train_images_normalized = augmented_train_images / 255.0
val_images_normalized = val_images / 255.0

# Create a simple CNN model
def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    return model

# Define input shape
input_shape = (128, 128, 3)  # Adjust according to your dataset

# Create model
model = create_model(input_shape)

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_images_normalized, train_labels, validation_data=(val_images_normalized, val_labels), epochs=24)

# Save the trained model in the recommended format
model.save('moss_detection_model.keras')
