import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Path to the template directory
TEMPLATE_DIR = "currency_templates/"

# Load template images
def load_templates():
    templates = {}
    for currency in os.listdir(TEMPLATE_DIR):
        currency_path = os.path.join(TEMPLATE_DIR, currency)
        if os.path.isdir(currency_path):
            front_image_path = os.path.join(currency_path, 'front.jpg')
            back_image_path = os.path.join(currency_path, 'back.jpg')
            
            if os.path.exists(front_image_path) and os.path.exists(back_image_path):
                templates[currency] = {
                    'front': cv2.imread(front_image_path, cv2.IMREAD_GRAYSCALE),
                    'back': cv2.imread(back_image_path, cv2.IMREAD_GRAYSCALE)
                }
    return templates

# Preprocess image
def preprocess_image(image, size=(600, 300)):
    if image is None:
        raise ValueError("Image could not be loaded; check the file path.")
    return cv2.resize(image, size)

# Compare images with SSIM
def compare_images(input_image, template_image):
    template_resized = cv2.resize(template_image, input_image.shape[::-1])
    score, _ = ssim(input_image, template_resized, full=True)
    return score

# Detect Watermark
def detect_watermark(image):
    _, thresholded = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return np.sum(thresholded == 255) > 1000  # Arbitrary threshold for watermark presence

# Detect Security Thread
def detect_security_thread(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 50), (180, 255, 150))
    return np.sum(mask) > 2000  # Threshold to confirm thread presence

# Detect Raised Print
def detect_raised_print(image):
    edges = cv2.Canny(image, 100, 200)
    return np.sum(edges) > 1000  # Arbitrary edge count threshold

# Main currency detection and fake note validation
def identify_currency_and_validate(input_front_image_path, input_back_image_path, similarity_threshold=0.7):
    input_front_image = cv2.imread(input_front_image_path, cv2.IMREAD_GRAYSCALE)
    input_back_image = cv2.imread(input_back_image_path, cv2.IMREAD_GRAYSCALE)
    
    if input_front_image is None or input_back_image is None:
        print("Error: Could not load one or both input images.")
        return None
    
    input_front_gray = preprocess_image(input_front_image)
    input_back_gray = preprocess_image(input_back_image)
    
    templates = load_templates()
    if not templates:
        print("No templates found in the specified directory.")
        return None
    
    match_scores = {}

    for currency, sides in templates.items():
        front_score = compare_images(input_front_gray, sides['front'])
        back_score = compare_images(input_back_gray, sides['back'])
        combined_score = (front_score + back_score) / 2
        match_scores[currency] = combined_score

    best_match = max(match_scores, key=match_scores.get)
    best_score = match_scores[best_match]
    
    if best_score > similarity_threshold:
        print(f"Detected Currency: {best_match} Rupees (Score: {best_score:.2f})")
        
        # Load color image for advanced feature detection
        input_front_color = cv2.imread(input_front_image_path)
        input_back_color = cv2.imread(input_back_image_path)
        
        # Perform fake money detection on the detected currency
        watermark_present = detect_watermark(input_front_gray)
        thread_present = detect_security_thread(input_front_color)
        raised_print_present = detect_raised_print(input_front_gray)
        
        # Verification - Confirm authenticity by checking each security feature
        if watermark_present and thread_present and raised_print_present:
            print("The note is authentic.")
        else:
            print("Warning: This note may be counterfeit.")
            if not watermark_present:
                print("Watermark not detected.")
            if not thread_present:
                print("Security thread not detected.")
            if not raised_print_present:
                print("Raised print not detected.")
        
        return best_match
    else:
        print("No accurate match found. Please try a clearer image.")
        return None

# Example usage
input_front_image_path = "uploaded_notes/front.jpg"  # Replace with the path to your input front image
input_back_image_path = "uploaded_notes/back.jpg"    # Replace with the path to your input back image
identify_currency_and_validate(input_front_image_path, input_back_image_path)
