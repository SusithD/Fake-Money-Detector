from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Set up paths and application configurations
TEMPLATE_DIR = "currency_templates/"
UPLOAD_FOLDER = 'uploaded_notes/'
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load currency templates for validation
def load_currency_templates():
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
            else:
                print(f"Warning: Missing image files for currency: {currency}")
    return templates

# Calculate similarity score between two images
def compare_images(input_image, template_image):
    if input_image.shape != template_image.shape:
        template_image = cv2.resize(template_image, input_image.shape[::-1])
    score, _ = ssim(input_image, template_image, full=True)
    return score

# Watermark detection with a specified threshold
def detect_watermark(image, template_front):
    front_resized = cv2.resize(template_front, (image.shape[1], image.shape[0]))
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    _, binary_template = cv2.threshold(front_resized, 127, 255, cv2.THRESH_BINARY)
    score, _ = ssim(binary_image, binary_template, full=True)
    return score > 0.80  # Threshold for watermark detection

# Microtext detection with density constraints
def detect_microtext(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    microtext_count = sum(1 for cnt in contours if 50 < cv2.contourArea(cnt) < 100)
    return 100 < microtext_count < 5000

# See-through feature detection
def detect_see_through(image):
    flipped_image = cv2.flip(image, 1)
    score, _ = ssim(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(flipped_image, cv2.COLOR_BGR2GRAY), full=True)
    return score < 0.90

# Security thread detection based on color consistency
def detect_security_thread(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask_green = cv2.inRange(hsv, (40, 100, 100), (80, 255, 255))
    combined_mask = cv2.bitwise_or(mask_red, mask_green)
    thread_intensity = np.sum(combined_mask) / 255
    return thread_intensity > 5000

# Intaglio print detection based on texture
def detect_intaglio_prints(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    _, binary = cv2.threshold(laplacian_abs, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    intaglio_count = sum(1 for cnt in contours if cv2.contourArea(cnt) > 100)
    return intaglio_count > 20

# Histogram comparison for color consistency
def histogram_similarity(img1, img2):
    hist_img1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist_img2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)

# Rotate Function
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


def identify_and_validate_currency(input_front_image_path, input_back_image_path, similarity_threshold=0.75):
    # Load and preprocess images
    input_front_image = cv2.imread(input_front_image_path, cv2.IMREAD_COLOR)
    input_back_image = cv2.imread(input_back_image_path, cv2.IMREAD_GRAYSCALE)

    if input_front_image is None or input_back_image is None:
        return "Error: Could not load one or both input images.", None

    templates = load_currency_templates()
    if not templates:
        return "No templates found in the specified directory.", None

    # Function to calculate match scores for the images against templates
    def calculate_match_scores(input_front, input_back):
        match_scores = {}
        for currency, sides in templates.items():
            front_score = compare_images(cv2.cvtColor(input_front, cv2.COLOR_BGR2GRAY), sides['front'])
            back_score = compare_images(input_back, sides['back'])
            combined_score = (front_score + back_score) / 2
            color_consistency = histogram_similarity(input_front, cv2.cvtColor(sides['front'], cv2.COLOR_GRAY2BGR))
            final_score = 0.7 * combined_score + 0.3 * color_consistency
            match_scores[currency] = final_score
        return match_scores

    # Step 1: Try matching without rotation
    match_scores = calculate_match_scores(input_front_image, input_back_image)
    best_match = max(match_scores, key=match_scores.get)
    best_score = match_scores[best_match]

    # Step 2: If initial match is below the threshold, rotate and retry
    if best_score < similarity_threshold:
        rotated_front_image = rotate_image(input_front_image)
        rotated_back_image = rotate_image(input_back_image)
        
        # Check match scores again with the rotated images
        match_scores = calculate_match_scores(rotated_front_image, rotated_back_image)
        best_match_rotated = max(match_scores, key=match_scores.get)
        best_score_rotated = match_scores[best_match_rotated]

        # If the rotated match is better, update the best match and use the rotated images
        if best_score_rotated > best_score:
            best_match = best_match_rotated
            best_score = best_score_rotated
            input_front_image = rotated_front_image  # Use the rotated image for further processing
            input_back_image = rotated_back_image      # Use the rotated image for further processing

    # Check if the best score meets the similarity threshold
    if best_score > similarity_threshold:
        front_template = templates[best_match]['front']
        
        # Perform other validations with the best matched (potentially rotated) images
        watermark_detected = detect_watermark(cv2.cvtColor(input_front_image, cv2.COLOR_BGR2GRAY), front_template)
        micro_text_present = detect_microtext(cv2.cvtColor(input_front_image, cv2.COLOR_BGR2GRAY))
        see_through_present = detect_see_through(input_front_image)
        security_thread_present = detect_security_thread(input_front_image)
        intaglio_prints_present = detect_intaglio_prints(cv2.cvtColor(input_front_image, cv2.COLOR_BGR2GRAY))

        # Calculate the feature score for authenticity
        feature_score = (
            0.25 * watermark_detected + 
            0.2 * micro_text_present +
            0.15 * see_through_present +
            0.1 * security_thread_present +
            0.3 * intaglio_prints_present
        )

        if feature_score > 0.85:
            authenticity_message = "The note is likely authentic."
        else:
            authenticity_message = "Warning: This note may be counterfeit."
            if not watermark_detected:
                authenticity_message += " Watermark not detected."
            if not micro_text_present:
                authenticity_message += " Micro-text not detected."
            if not see_through_present:
                authenticity_message += " See-through feature not detected."
            if not security_thread_present:
                authenticity_message += " Security thread not detected."
            if not intaglio_prints_present:
                authenticity_message += " Intaglio prints not detected."

        return best_match, authenticity_message
    else:
        return None, "No accurate match found. Please try a clearer image."


# Flask route definitions
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'front' not in request.files or 'back' not in request.files:
        error_message = 'Please upload both front and back images.'
        return render_template('index.html', authenticity_message=error_message)

    front_image = request.files['front']
    back_image = request.files['back']
    
    if not front_image or not back_image:
        error_message = 'Both images are required.'
        return render_template('index.html', authenticity_message=error_message)
    
    if not front_image.filename.lower().endswith(('.png', '.jpg', '.jpeg')) or \
       not back_image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        error_message = 'Only image files (.jpg, .jpeg, .png) are allowed.'
        return render_template('index.html', authenticity_message=error_message)
    
    front_path = os.path.join(app.config['UPLOAD_FOLDER'], front_image.filename)
    back_path = os.path.join(app.config['UPLOAD_FOLDER'], back_image.filename)
    
    front_image.save(front_path)
    back_image.save(back_path)

    result, authenticity_message = identify_and_validate_currency(front_path, back_path)

    return render_template('index.html', result=result, authenticity_message=authenticity_message)

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
