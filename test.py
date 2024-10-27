import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotate_image(image):
    # Get the dimensions of the image
    (h, w) = image.shape[:2]
    
    # Define the center of the image
    center = (w // 2, h // 2)
    
    # Create a rotation matrix for 180 degrees
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    
    # Apply the affine transformation
    rotated_image = cv2.warpAffine(image, M, (w, h))
    
    return rotated_image

# Load an image
image_path = "uploaded_notes/back 20.jpg"  # Update this path
image = cv2.imread(image_path)

# Rotate the image
rotated_image = rotate_image(image)

# Convert BGR to RGB for correct color display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)

# Display the original and rotated images using matplotlib
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")  # Turn off axis numbers and ticks

# Rotated Image
plt.subplot(1, 2, 2)
plt.imshow(rotated_image_rgb)
plt.title("Rotated Image")
plt.axis("off")  # Turn off axis numbers and ticks

# Show the plot
plt.tight_layout()
plt.show()
