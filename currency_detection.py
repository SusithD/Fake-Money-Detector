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
                    'back': cv2.imread(back_image_path, cv2.IMREAD_GRAYSCALE),
                }
    return templates

# Enhanced watermark detection with stricter thresholds
def detect_watermark(image, template_front):
    front_resized = cv2.resize(template_front, (image.shape[1], image.shape[0]))
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    _, binary_template = cv2.threshold(front_resized, 127, 255, cv2.THRESH_BINARY)
    score, _ = ssim(binary_image, binary_template, full=True)
    print(f"Watermark SSIM Score: {score:.2f}")
    return score > 0.80  # Increased threshold for robustness

# Microtext detection with density check
def detect_micro_text(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    microtext_count = sum(1 for cnt in contours if 50 < cv2.contourArea(cnt) < 100)  # Adjusted density range
    print(f"Microtext Count: {microtext_count}")
    return 100 < microtext_count < 5000  # Narrowed acceptable range for microtext density

# See-Through Feature Detection
def detect_see_through(image):
    flipped_image = cv2.flip(image, 1)
    score, _ = ssim(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(flipped_image, cv2.COLOR_BGR2GRAY), full=True)
    print(f"See-Through Feature SSIM Score: {score:.2f}")
    return score < 0.90  # Higher score indicates misalignment or absence

# Security Thread Detection
def detect_security_thread(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))  # Detecting red
    mask_green = cv2.inRange(hsv, (40, 100, 100), (80, 255, 255))  # Detecting green
    combined_mask = cv2.bitwise_or(mask_red, mask_green)
    thread_intensity = np.sum(combined_mask) / 255
    print(f"Security Thread Intensity: {thread_intensity}")
    return thread_intensity > 5000  # Threshold for detecting the security thread

# Intaglio Prints Detection
def detect_intaglio_prints(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    _, binary = cv2.threshold(laplacian_abs, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    intaglio_count = sum(1 for cnt in contours if cv2.contourArea(cnt) > 100)  # Adjust based on expected texture
    print(f"Intaglio Prints Count: {intaglio_count}")
    return intaglio_count > 20  # Minimum count for detecting intaglio prints

# Template matching with histogram analysis for color consistency
def compare_images(input_image, template_image):
    template_resized = cv2.resize(template_image, input_image.shape[::-1])
    score, _ = ssim(input_image, template_resized, full=True)
    print(f"SSIM Score: {score:.2f}")
    return score

# Histogram comparison for color consistency
def histogram_similarity(img1, img2):
    hist_img1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist_img2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)

# Main function for currency identification and validation
def identify_currency_and_validate(input_front_image_path, input_back_image_path, similarity_threshold=0.75):
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
        color_consistency = histogram_similarity(input_front_image, cv2.cvtColor(sides['front'], cv2.COLOR_GRAY2BGR))
        final_score = 0.7 * combined_score + 0.3 * color_consistency
        match_scores[currency] = final_score

    best_match = max(match_scores, key=match_scores.get)
    best_score = match_scores[best_match]
    
    print(f"Best Match: {best_match}, Score: {best_score:.2f}")

    if best_score > similarity_threshold:
        print(f"Detected Currency: {best_match} (Score: {best_score:.2f})")
        front_template = templates[best_match]['front']
        
        # Feature detections with stricter thresholds
        watermark_detected = detect_watermark(cv2.cvtColor(input_front_image, cv2.COLOR_BGR2GRAY), front_template)
        micro_text_present = detect_micro_text(cv2.cvtColor(input_front_image, cv2.COLOR_BGR2GRAY))
        see_through_present = detect_see_through(input_front_image)
        security_thread_present = detect_security_thread(input_front_image)
        intaglio_prints_present = detect_intaglio_prints(cv2.cvtColor(input_front_image, cv2.COLOR_BGR2GRAY))
        
        # Scoring and authenticity check with weighted thresholds
        feature_score = (
            0.25 * watermark_detected + 
            0.2 * micro_text_present +
            0.15 * see_through_present +
            0.1 * security_thread_present +
            0.3 * intaglio_prints_present
        )

        print(f"Feature Score: {feature_score:.2f}")

        if feature_score > 0.85:
            print("The note is likely authentic.")
        else:
            print("Warning: This note may be counterfeit.")
            if not watermark_detected:
                print("Watermark not detected.")
            if not micro_text_present:
                print("Micro-text not detected.")
            if not see_through_present:
                print("See-through feature not detected.")
            if not security_thread_present:
                print("Security thread not detected.")
            if not intaglio_prints_present:
                print("Intaglio prints not detected.")

        return best_match
    else:
        print("No accurate match found. Please try a clearer image.")
        return None

# Example usage
input_front_image_path = "uploaded_notes/front 100.jpg"
input_back_image_path = "uploaded_notes/back 50.jpg"
identify_currency_and_validate(input_front_image_path, input_back_image_path)