import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Path to the template directory
TEMPLATE_DIR = "currency_templates/"

# Function to load template images
def load_templates():
    templates = {}
    for currency in os.listdir(TEMPLATE_DIR):
        currency_path = os.path.join(TEMPLATE_DIR, currency)
        if os.path.isdir(currency_path):
            # Define exact paths for front and back images
            front_image_path = os.path.join(currency_path, 'front.jpg')
            back_image_path = os.path.join(currency_path, 'back.jpg')
            
            # Load images if they exist
            if os.path.exists(front_image_path) and os.path.exists(back_image_path):
                templates[currency] = {
                    'front': cv2.imread(front_image_path, cv2.IMREAD_GRAYSCALE),
                    'back': cv2.imread(back_image_path, cv2.IMREAD_GRAYSCALE)
                }
    return templates

# Function to preprocess images (resize to consistent size)
def preprocess_image(image, size=(600, 300)):
    if image is None:
        raise ValueError("Image could not be loaded; check the file path.")
    return cv2.resize(image, size)  # Resize for consistency

# Function to compare images using Structural Similarity Index (SSIM)
def compare_images(input_image, template_image):
    # Resize template to match input image size
    template_resized = cv2.resize(template_image, input_image.shape[::-1])
    score, _ = ssim(input_image, template_resized, full=True)
    return score

# Main function to identify currency note
def identify_currency(input_front_image_path, input_back_image_path, similarity_threshold=0.7):
    # Load and preprocess both input images
    input_front_image = cv2.imread(input_front_image_path, cv2.IMREAD_GRAYSCALE)
    input_back_image = cv2.imread(input_back_image_path, cv2.IMREAD_GRAYSCALE)

    # Verify images loaded correctly
    if input_front_image is None:
        print(f"Error: Could not load front image at {input_front_image_path}")
        return None
    if input_back_image is None:
        print(f"Error: Could not load back image at {input_back_image_path}")
        return None

    # Resize images to consistent size
    input_front_gray = preprocess_image(input_front_image)
    input_back_gray = preprocess_image(input_back_image)
    
    # Load currency templates
    templates = load_templates()
    if not templates:
        print("No templates found in the specified directory.")
        return None
    
    # Dictionary to store match scores for each template
    match_scores = {}

    # Compare the input images with each template (front and back)
    for currency, sides in templates.items():
        front_score = compare_images(input_front_gray, sides['front'])
        back_score = compare_images(input_back_gray, sides['back'])
        
        # Combine scores, higher combined score implies a better match
        combined_score = (front_score + back_score) / 2
        match_scores[currency] = combined_score

    # Find the best match based on the highest combined score
    best_match = max(match_scores, key=match_scores.get)
    best_score = match_scores[best_match]
    
    # Print or return results based on similarity threshold
    if best_score > similarity_threshold:
        print(f"Best match: {best_match} Rupees (Combined Score: {best_score:.2f})")
        return best_match
    else:
        print("No accurate match found. Please try a clearer image.")
        return None

# Example usage
input_front_image_path = "uploaded_notes/front_currency.jpg"  # Replace with path to your input front image
input_back_image_path = "uploaded_notes/back_currency.jpg"    # Replace with path to your input back image
identify_currency(input_front_image_path, input_back_image_path)
