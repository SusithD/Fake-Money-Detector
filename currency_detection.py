import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ORB detector for feature matching
orb = cv2.ORB_create()

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

# Color Consistency Check
def check_color_consistency(image, expected_hue_range=(30, 90)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (expected_hue_range[0], 50, 50), (expected_hue_range[1], 255, 255))
    return np.sum(mask) > 5000  # Threshold for expected color presence

# Micro-Text Detection
def detect_micro_text(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return np.sum(edges) > 1500  # Threshold for high-density edges indicating micro-text

# Hologram/Security Foil Detection
def detect_hologram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    reflection_mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
    return np.sum(reflection_mask) > 3000  # Threshold for reflective hologram areas

# Image Statistical Distribution Analysis
def analyze_image_distribution(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Check for anomalies in the histogram
    return np.var(hist) < 1000  # Adjust threshold as necessary

# ORB feature matching
def orb_match(input_image, template_image):
    kp1, des1 = orb.detectAndCompute(input_image, None)
    kp2, des2 = orb.detectAndCompute(template_image, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return len(matches) / min(len(kp1), len(kp2)) > 0.2

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
        
        input_front_color = cv2.imread(input_front_image_path)
        
        # Perform fake money detection on the detected currency
        color_consistency = check_color_consistency(input_front_color)
        micro_text_present = detect_micro_text(input_front_gray)
        hologram_present = detect_hologram(input_front_color)
        image_distribution_normal = analyze_image_distribution(input_front_color)
        orb_feature_match = orb_match(input_front_gray, templates[best_match]['front'])
        
        # Combine checks for final verification
        if all([color_consistency, micro_text_present, hologram_present, image_distribution_normal, orb_feature_match]):
            print("The note is likely authentic.")
        else:
            print("Warning: This note may be counterfeit.")
            if not color_consistency:
                print("Color consistency check failed.")
            if not micro_text_present:
                print("Micro-text not detected.")
            if not hologram_present:
                print("Hologram not detected.")
            if not image_distribution_normal:
                print("Image distribution check failed.")
            if not orb_feature_match:
                print("Key features not matched.")
        
        return best_match
    else:
        print("No accurate match found. Please try a clearer image.")
        return None

# Example usage
input_front_image_path = "uploaded_notes/front.jpg"
input_back_image_path = "uploaded_notes/back.jpg"
identify_currency_and_validate(input_front_image_path, input_back_image_path)
