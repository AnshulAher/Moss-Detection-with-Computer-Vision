import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from keras import layers, models

# Function to load images and labels from folder
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = cv2.imread(img_path)
        if image is not None:
            images.append(image)
            labels.append(label)
    return images, labels

# Load images from moss folder
moss_images, moss_labels = load_images_from_folder("C:\Brian\Research\omossimg2", label=1)
# Load images from non-moss folder
non_moss_images, non_moss_labels = load_images_from_folder("C:\Brian\Research\onmossimg2", label=0)

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

# Define input shape
input_shape = (128, 128, 3)  # Adjust according to your dataset

# Create model
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

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10)

# Save the trained model using the native Keras format
model.save('moss_detection_model.keras')

print(os.getcwd())