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
            watermark_image_path = os.path.join(currency_path, 'watermark.jpg')  # Load watermark

            if (os.path.exists(front_image_path) and 
                os.path.exists(back_image_path) and 
                os.path.exists(watermark_image_path)):
                templates[currency] = {
                    'front': cv2.imread(front_image_path, cv2.IMREAD_GRAYSCALE),
                    'back': cv2.imread(back_image_path, cv2.IMREAD_GRAYSCALE),
                    'watermark': cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)  # Add watermark
                }
    return templates

# Watermark Detection
def detect_watermark(image, template_watermark):
    # Resize the watermark template for comparison
    watermark_resized = cv2.resize(template_watermark, (image.shape[1] // 3, image.shape[0] // 3))
    result = cv2.matchTemplate(image, watermark_resized, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7  # Adjust this threshold as needed
    loc = np.where(result >= threshold)
    return len(loc[0]) > 0  # Return True if watermark is detected

# Microtext Detection
def detect_micro_text(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return np.sum(edges) > 1500  # Threshold for high-density edges indicating microtext

# Hologram/Security Foil Detection
def detect_hologram(image):
    # Convert to color if the input is grayscale
    if len(image.shape) == 2:  # Check if it's a grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    reflection_mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
    return np.sum(reflection_mask) > 3000  # Threshold for reflective hologram areas

# Edge Detection for Fine Lines
def detect_fine_lines(image):
    edges = cv2.Canny(image, 100, 200)
    return np.sum(edges) > 2000  # Adjusted threshold for fine lines

# Template matching using SSIM
def compare_images(input_image, template_image):
    template_resized = cv2.resize(template_image, input_image.shape[::-1])
    score, _ = ssim(input_image, template_resized, full=True)
    return score

# Main currency detection and fake note validation
def identify_currency_and_validate(input_front_image_path, input_back_image_path, similarity_threshold=0.75):
    # Load input images in color for hologram detection
    input_front_image = cv2.imread(input_front_image_path, cv2.IMREAD_COLOR)
    input_back_image = cv2.imread(input_back_image_path, cv2.IMREAD_GRAYSCALE)
    
    if input_front_image is None or input_back_image is None:
        print("Error: Could not load one or both input images.")
        return None
    
    templates = load_templates()
    if not templates:
        print("No templates found in the specified directory.")
        return None
    
    match_scores = {}

    for currency, sides in templates.items():
        front_score = compare_images(cv2.cvtColor(input_front_image, cv2.COLOR_BGR2GRAY), sides['front'])
        back_score = compare_images(input_back_image, sides['back'])
        combined_score = (front_score + back_score) / 2
        match_scores[currency] = combined_score

    best_match = max(match_scores, key=match_scores.get)
    best_score = match_scores[best_match]
    
    print(f"Best Match: {best_match}, Score: {best_score:.2f}")

    if best_score > similarity_threshold:
        print(f"Detected Currency: {best_match} (Score: {best_score:.2f})")

        # Get the watermark template for the matched currency
        watermark_template = templates[best_match]['watermark']

        # Check for features
        watermark_detected = detect_watermark(cv2.cvtColor(input_front_image, cv2.COLOR_BGR2GRAY), watermark_template)
        micro_text_present = detect_micro_text(cv2.cvtColor(input_front_image, cv2.COLOR_BGR2GRAY))
        hologram_present = detect_hologram(input_front_image)
        fine_lines_present = detect_fine_lines(cv2.cvtColor(input_front_image, cv2.COLOR_BGR2GRAY))
        
        # Debugging information
        print(f"Watermark Detected: {watermark_detected}, Microtext Present: {micro_text_present}, Hologram Present: {hologram_present}, Fine Lines Detected: {fine_lines_present}")

        # Combine checks for final verification with weighted scoring
        score = (0.25 * watermark_detected +
                 0.25 * micro_text_present +
                 0.25 * hologram_present +
                 0.25 * fine_lines_present)

        print(f"Final Score: {score:.2f}")  # Debugging line to log the final score

        if score > 0.75:  # Threshold for authenticity
            print("The note is likely authentic.")
        else:
            print("Warning: This note may be counterfeit.")
            if not watermark_detected:
                print("Watermark not detected.")
            if not micro_text_present:
                print("Micro-text not detected.")
            if not hologram_present:
                print("Hologram not detected.")
            if not fine_lines_present:
                print("Fine lines not detected.")

        return best_match
    else:
        print("No accurate match found. Please try a clearer image.")
        return None

# Example usage
input_front_image_path = "uploaded_notes/front.jpg"
input_back_image_path = "uploaded_notes/back.jpg"
identify_currency_and_validate(input_front_image_path, input_back_image_path)
