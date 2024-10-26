import cv2
import numpy as np
import pytesseract
import os
import sys
from matplotlib import pyplot as plt

# ===========================
# Step 1: Define Directory Paths
# ===========================

BASE_DIR = os.path.dirname(os.path.abspath(_file_))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'currency_templates')
UPLOADED_NOTES_DIR = os.path.join(BASE_DIR, 'uploaded_notes')

# ===========================
# Step 2: Preprocessing Functions
# ===========================

def preprocess_image(image, display_steps=False):
    """
    Preprocess the input image for feature detection, ensuring it's in grayscale.
    Optionally display each preprocessing step.
    
    Args:
        image (numpy.ndarray): Input image.
        display_steps (bool): If True, display images at each preprocessing step.
    
    Returns:
        numpy.ndarray: Preprocessed grayscale image.
    """
    if image is None:
        raise ValueError("Input image is None.")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if display_steps:
            plt.figure(figsize=(10, 8))
            plt.imshow(gray, cmap='gray')
            plt.title("Grayscale Image")
            plt.axis('off')
            plt.show()
    else:
        gray = image.copy()
        if display_steps:
            plt.figure(figsize=(10, 8))
            plt.imshow(gray, cmap='gray')
            plt.title("Original Grayscale Image")
            plt.axis('off')
            plt.show()
    
    # Apply Bilateral Filter to reduce noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    if display_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(gray, cmap='gray')
        plt.title("After Bilateral Filter")
        plt.axis('off')
        plt.show()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    if display_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(gray, cmap='gray')
        plt.title("After CLAHE")
        plt.axis('off')
        plt.show()
    
    # Apply Sharpening Filter
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    if display_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(gray, cmap='gray')
        plt.title("After Sharpening")
        plt.axis('off')
        plt.show()
    
    return gray

def preprocess_for_blind_recognition(image, display_steps=False):
    """
    Preprocess the input image specifically for blind recognition feature detection.
    Optionally display the binary image.
    
    Args:
        image (numpy.ndarray): Input grayscale image.
        display_steps (bool): If True, display the thresholded image.
    
    Returns:
        numpy.ndarray: Thresholded binary image.
    """
    if image is None:
        raise ValueError("Input image is None.")
    
    # Apply Gaussian Blur to reduce noise
    gray = cv2.GaussianBlur(image, (5, 5), 0)
    if display_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(gray, cmap='gray')
        plt.title("After Gaussian Blur")
        plt.axis('off')
        plt.show()
    
    # Thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    if display_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(thresh, cmap='gray')
        plt.title("After Thresholding for Blind Recognition")
        plt.axis('off')
        plt.show()
    
    return thresh

# ===========================
# Step 3: Load Template Images
# ===========================

def load_templates():
    """
    Load currency templates from the TEMPLATE_DIR.
    
    Returns:
        dict: Nested dictionary with currency denominations as keys and their respective templates.
    """
    currency_notes = ['lkr20', 'lkr50', 'lkr100', 'lkr500', 'lkr1000', 'lkr5000']
    template_features = [
        'front.jpg',
        'back.jpg',
        'watermark_front.jpg',
        'security_thread_front.jpg',
        'cornerstone.jpg',
        'see_through1.jpg',
        'see_through2.jpg',
        'see_through3.jpg',
        'extra_small_text_front.jpg',
        'blind_recognition_feature_front.jpg',
        'raised_paint_front.jpg'
    ]
    
    templates = {}
    
    for note in currency_notes:
        templates[note] = {}
        note_dir = os.path.join(TEMPLATE_DIR, note)
        
        for feature in template_features:
            feature_path = os.path.join(note_dir, feature)
            if not os.path.exists(feature_path):
                print(f"Warning: {feature} not found in {note_dir}. Skipping this feature.")
                templates[note][feature.split('.')[0]] = None
                continue
            
            # Load the image in grayscale
            template_image = cv2.imread(feature_path, cv2.IMREAD_GRAYSCALE)
            if template_image is None:
                print(f"Error: Failed to load {feature} from {note_dir}.")
                templates[note][feature.split('.')[0]] = None
                continue
            
            # Optionally, preprocess the template image for consistency
            template_image = preprocess_image(template_image, display_steps=False)
            templates[note][feature.split('.')[0]] = template_image
    
    print("Template images loaded successfully.")
    return templates

# ===========================
# Step 4: Load Uploaded Note Images
# ===========================

def load_uploaded_images(display_steps=False):
    """
    Load and preprocess the uploaded currency note images.
    
    Args:
        display_steps (bool): If True, display preprocessing steps.
    
    Returns:
        tuple: Preprocessed grayscale images of the front and back notes.
    """
    front_image_path = os.path.join(UPLOADED_NOTES_DIR, 'front_currency.jpg')
    back_image_path = os.path.join(UPLOADED_NOTES_DIR, 'back_currency.jpg')
    
    front_image = cv2.imread(front_image_path)
    back_image = cv2.imread(back_image_path)
    
    if front_image is None:
        raise IOError(f"Uploaded front image not found or unreadable at {front_image_path}.")
    if back_image is None:
        raise IOError(f"Uploaded back image not found or unreadable at {back_image_path}.")
    
    # Preprocess images with optional display
    front_image_gray = preprocess_image(front_image, display_steps=display_steps)
    back_image_gray = preprocess_image(back_image, display_steps=display_steps)
    
    return front_image_gray, back_image_gray

# ===========================
# Step 5: Image Alignment
# ===========================

def align_images(image1, image2, display_steps=False):
    """
    Align image2 to image1 using ORB keypoint matching.
    
    Args:
        image1 (numpy.ndarray): Reference grayscale image.
        image2 (numpy.ndarray): Grayscale image to be aligned.
        display_steps (bool): If True, display keypoints and matches.
    
    Returns:
        numpy.ndarray: Aligned version of image2.
    """
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # Use BFMatcher with Hamming distance and crossCheck
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    if des1 is None or des2 is None:
        print("Warning: No descriptors found. Skipping alignment.")
        return image2
    
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Use top 10 matches for homography
    good_matches = matches[:10]
    
    if len(good_matches) < 4:
        print("Warning: Not enough matches for alignment.")
        return image2
    
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
    
    for i, match in enumerate(good_matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    
    # Compute homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    if h is None:
        print("Warning: Homography could not be computed.")
        return image2
    
    height, width = image1.shape
    aligned_image = cv2.warpPerspective(image2, h, (width, height))
    
    if display_steps:
        # Draw matches
        matched_image = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(15, 10))
        plt.imshow(matched_image, cmap='gray')
        plt.title("ORB Keypoint Matches")
        plt.axis('off')
        plt.show()
    
    return aligned_image

# ===========================
# Step 6: Feature Detection Functions
# ===========================

def rotate_image(image, angle):
    """
    Rotate an image around its center without cropping.

    Args:
        image (numpy.ndarray): Grayscale image to rotate.
        angle (float): Angle in degrees to rotate the image.

    Returns:
        numpy.ndarray: Rotated image.
    """
    (h, w) = image.shape
    center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the sine and cosine (i.e., the rotation components of the matrix)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    # Compute new bounding dimensions of the image
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]

    # Perform the actual rotation and return the image
    rotated = cv2.warpAffine(image, M, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def match_template_multi_scale_rotation(note_image, template, scales=[0.8, 1.0, 1.2], angles=[-15, -10, -5, 0, 5, 10, 15], display_steps=False):
    """
    Perform template matching with multiple scales and rotations to detect features.

    Args:
        note_image (numpy.ndarray): Preprocessed grayscale image of the note.
        template (numpy.ndarray or None): Grayscale template image for the feature.
        scales (list): List of scales to resize the template.
        angles (list): List of angles to rotate the template.
        display_steps (bool): If True, display the matching result.

    Returns:
        tuple: (bool indicating detection, max_score, best_loc, best_scale, best_angle)
    """
    if template is None:
        return False, 0, None, None, None

    max_score = 0
    detected = False
    best_loc = None
    best_scale = None
    best_angle = None

    for scale in scales:
        # Resize the template
        try:
            resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"Error resizing template: {e}")
            continue

        if resized_template.shape[0] > note_image.shape[0] or resized_template.shape[1] > note_image.shape[1]:
            continue  # Skip if template is larger than the note image

        for angle in angles:
            # Rotate the resized template
            rotated_template = rotate_image(resized_template, angle)

            # Perform template matching
            result = cv2.matchTemplate(note_image, rotated_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # Update maximum score and best parameters if a higher score is found
            if max_val > max_score:
                max_score = max_val
                best_loc = max_loc
                best_scale = scale
                best_angle = angle

    if max_score >= 0.8:  # Threshold can be adjusted based on testing
        detected = True

    if display_steps and detected:
        # Draw rectangle around the detected area
        top_left = best_loc
        rotated_resized_template = rotate_image(cv2.resize(template, None, fx=best_scale, fy=best_scale, interpolation=cv2.INTER_AREA), best_angle)
        h, w = rotated_resized_template.shape
        display_image = note_image.copy()
        cv2.rectangle(display_image, top_left, (top_left[0] + w, top_left[1] + h), 255, 2)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(display_image, cmap='gray')
        plt.title(f"Feature Detected at Scale: {best_scale}, Angle: {best_angle}")
        plt.axis('off')
        plt.show()

    return detected, max_score, best_loc, best_scale, best_angle

def detect_watermark(note_image, template, display_steps=False):
    """
    Detect watermark by matching the template with the note image using multi-scale and rotation-invariant matching.

    Args:
        note_image (numpy.ndarray): Preprocessed grayscale image of the note.
        template (numpy.ndarray or None): Grayscale template image for watermark.
        display_steps (bool): If True, display the matching result.

    Returns:
        tuple: (bool indicating detection, matching score, best_loc, best_scale, best_angle)
    """
    detected, max_score, best_loc, best_scale, best_angle = match_template_multi_scale_rotation(
        note_image, template, scales=[0.8, 1.0, 1.2], angles=[-15, -10, -5, 0, 5, 10, 15], display_steps=display_steps
    )
    return detected, max_score, best_loc, best_scale, best_angle

def detect_cornerstone(note_image, template, display_steps=False):
    """
    Detect cornerstone feature using multi-scale and rotation-invariant template matching.

    Args:
        note_image (numpy.ndarray): Preprocessed grayscale image of the note.
        template (numpy.ndarray or None): Grayscale template image for cornerstone.
        display_steps (bool): If True, display the matching result.

    Returns:
        tuple: (bool indicating detection, matching score, best_loc, best_scale, best_angle)
    """
    detected, max_score, best_loc, best_scale, best_angle = match_template_multi_scale_rotation(
        note_image, template, scales=[0.8, 1.0, 1.2], angles=[-15, -10, -5, 0, 5, 10, 15], display_steps=display_steps
    )
    return detected, max_score, best_loc, best_scale, best_angle

def detect_security_thread(note_image, display_steps=False):
    """
    Detect the security thread in the note image using edge detection.

    Args:
        note_image (numpy.ndarray): Preprocessed grayscale image of the note.
        display_steps (bool): If True, display the edge image.

    Returns:
        bool: True if security thread detected, else False.
    """
    edges = cv2.Canny(note_image, 50, 150)
    if display_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(edges, cmap='gray')
        plt.title("Canny Edges for Security Thread Detection")
        plt.axis('off')
        plt.show()
    
    # Sum of edges indicates possible security thread presence
    detected = np.sum(edges) > 1000  # Threshold can be adjusted based on testing
    return detected

def detect_extra_small_text(note_image, display_steps=False):
    """
    Detect extra small text in the note image using OCR.

    Args:
        note_image (numpy.ndarray): Preprocessed grayscale image of the note.
        display_steps (bool): If True, display the detected text.

    Returns:
        bool: True if any extra small text is detected, else False.
    """
    text = pytesseract.image_to_string(note_image, config='--psm 6')
    
    if display_steps:
        print(f"Detected Text: {text.strip()}")
    
    return text.strip() != ""

def detect_blind_recognition(note_image, template, display_steps=False):
    """
    Detect blind recognition feature (e.g., printed dots).

    Args:
        note_image (numpy.ndarray): Preprocessed grayscale image of the note.
        template (numpy.ndarray or None): Grayscale template image for blind recognition.
        display_steps (bool): If True, display the thresholded image and contours.

    Returns:
        tuple: (bool indicating detection, match score)
    """
    if template is None:
        return False, 0  # Return a score of 0 if the template is None
    
    # Preprocess images for blind recognition
    note_image_thresh = preprocess_for_blind_recognition(note_image, display_steps=display_steps)
    template_thresh = preprocess_for_blind_recognition(template, display_steps=display_steps)
    
    # Find contours in the uploaded image
    contours_note, _ = cv2.findContours(note_image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dot_contours_note = [cnt for cnt in contours_note if 5 < cv2.contourArea(cnt) < 50]
    
    # Find contours in the template
    contours_template, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dot_contours_template = [cnt for cnt in contours_template if 5 < cv2.contourArea(cnt) < 50]
    
    # Calculate match score based on the number of matching dots
    if len(dot_contours_note) == 0 or len(dot_contours_template) == 0:
        match_score = 0
    else:
        match_score = min(len(dot_contours_note), len(dot_contours_template)) / max(len(dot_contours_note), len(dot_contours_template))
    
    detected = match_score > 0.8
    
    if display_steps:
        # Draw contours on the note image
        note_contours_image = cv2.cvtColor(note_image_thresh, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(note_contours_image, dot_contours_note, -1, (0, 255, 0), 1)
        plt.figure(figsize=(10, 8))
        plt.imshow(note_contours_image)
        plt.title("Detected Dots for Blind Recognition")
        plt.axis('off')
        plt.show()
        
        # Print match score
        print(f"Blind Recognition Match Score: {match_score:.2f}")
    
    return detected, match_score

def detect_raised_paint(note_image, display_steps=False):
    """
    Detect raised paint areas using texture analysis.

    Args:
        note_image (numpy.ndarray): Preprocessed grayscale image of the note.
        display_steps (bool): If True, display the Laplacian variance.

    Returns:
        bool: True if raised paint detected, else False.
    """
    laplacian_var = cv2.Laplacian(note_image, cv2.CV_64F).var()
    
    if display_steps:
        print(f"Laplacian Variance for Raised Paint Detection: {laplacian_var:.2f}")
    
    detected = laplacian_var > 100  # Threshold for detecting raised textures
    return detected

def detect_see_through(note_image_front, note_image_back, templates, display_steps=False):
    """
    Detect see-through features by comparing front and back images against see-through templates.

    Args:
        note_image_front (numpy.ndarray): Preprocessed grayscale image of the front note.
        note_image_back (numpy.ndarray): Preprocessed grayscale image of the back note.
        templates (dict): Dictionary of templates for the current currency.
        display_steps (bool): If True, display the matching results.

    Returns:
        bool: True if see-through feature detected, else False.
    """
    for i in range(1, 4):
        template = templates.get(f'see_through{i}', None)
        if template is not None:
            # Match with front image
            result_front = cv2.matchTemplate(note_image_front, template, cv2.TM_CCOEFF_NORMED)
            max_val_front = np.max(result_front)
            
            # Match with back image
            result_back = cv2.matchTemplate(note_image_back, template, cv2.TM_CCOEFF_NORMED)
            max_val_back = np.max(result_back)
            
            if display_steps:
                # Display front match
                plt.figure(figsize=(10, 8))
                plt.imshow(result_front, cmap='gray')
                plt.title(f"See-Through {i} Front Match Score: {max_val_front:.2f}")
                plt.colorbar()
                plt.axis('off')
                plt.show()
                
                # Display back match
                plt.figure(figsize=(10, 8))
                plt.imshow(result_back, cmap='gray')
                plt.title(f"See-Through {i} Back Match Score: {max_val_back:.2f}")
                plt.colorbar()
                plt.axis('off')
                plt.show()
            
            if max_val_front >= 0.8 or max_val_back >= 0.8:
                return True
    
    return False

# ===========================
# Step 7: Feature Detection Logic for Front and Back
# ===========================

def detect_features(templates, uploaded_note_images, display_steps=False):
    """
    Detect features in the uploaded note images (both front and back) by comparing them with the templates.

    Args:
        templates (dict): Nested dictionary with currency templates.
        uploaded_note_images (tuple): Preprocessed grayscale images of the front and back notes.
        display_steps (bool): If True, display preprocessing and detection steps.

    Returns:
        dict: Nested dictionary with detection results for each currency.
    """
    front_image, back_image = uploaded_note_images
    results = {}
    
    for note_name, note_templates in templates.items():
        match_data = {}
        
        print(f"\nProcessing {note_name.upper()}...")
        
        # Detect watermark in front image
        watermark_detected, watermark_score, _, _, _ = detect_watermark(
            front_image, note_templates.get('watermark_front'), display_steps=display_steps
        )
        match_data['watermark'] = watermark_detected
        print(f" - Watermark Detected: {watermark_detected} (Score: {watermark_score:.2f})")
        
        # Detect security thread in front image
        security_thread_detected = detect_security_thread(front_image, display_steps=display_steps)
        match_data['security_thread'] = security_thread_detected
        print(f" - Security Thread Detected: {security_thread_detected}")
        
        # Detect cornerstone feature in front image
        cornerstone_detected, cornerstone_score, _, _, _ = detect_cornerstone(
            front_image, note_templates.get('cornerstone'), display_steps=display_steps
        )
        match_data['cornerstone'] = cornerstone_detected
        print(f" - Cornerstone Detected: {cornerstone_detected} (Score: {cornerstone_score:.2f})")
        
        # Detect extra small text in front image
        extra_small_text_detected = detect_extra_small_text(front_image, display_steps=display_steps)
        match_data['extra_small_text'] = extra_small_text_detected
        print(f" - Extra Small Text Detected: {extra_small_text_detected}")
        
        # Detect blind recognition feature in front image
        blind_recognition_detected, blind_recognition_score = detect_blind_recognition(
            front_image, note_templates.get('blind_recognition_feature_front'), display_steps=display_steps
        )
        match_data['blind_recognition_feature'] = blind_recognition_detected
        print(f" - Blind Recognition Feature Detected: {blind_recognition_detected} (Score: {blind_recognition_score:.2f})")
        
        # Detect raised paint in front image
        raised_paint_detected = detect_raised_paint(front_image, display_steps=display_steps)
        match_data['raised_paint'] = raised_paint_detected
        print(f" - Raised Paint Detected: {raised_paint_detected}")
        
        # Detect see-through features
        see_through_detected = detect_see_through(front_image, back_image, note_templates, display_steps=display_steps)
        match_data['see_through'] = see_through_detected
        print(f" - See-Through Feature Detected: {see_through_detected}")
        
        # Compare entire front and back images with their templates
        front_match, front_score, _, _, _ = match_template_multi_scale_rotation(
            front_image, note_templates.get('front'), display_steps=display_steps
        )
        back_match, back_score, _, _, _ = match_template_multi_scale_rotation(
            back_image, note_templates.get('back'), display_steps=display_steps
        )
        match_data['front_match'] = front_match
        match_data['back_match'] = back_match
        print(f" - Front Image Match: {front_match} (Score: {front_score:.2f})")
        print(f" - Back Image Match: {back_match} (Score: {back_score:.2f})")
        
        results[note_name] = match_data
    
    return results

# ===========================
# Step 8: Identify Currency Based on Results
# ===========================

def identify_currency(results):
    """
    Identify which currency note matches best based on front and back image matches and feature detections.

    Args:
        results (dict): Nested dictionary with detection results for each currency.

    Returns:
        str: Identified currency denomination or a message indicating failure.
    """
    identified_note = None
    max_score = 0

    for note_name, match_data in results.items():
        # Assign weights to different features for scoring
        score = 0
        if match_data.get('watermark'):
            score += 3
        if match_data.get('security_thread'):
            score += 3
        if match_data.get('cornerstone'):
            score += 2
        if match_data.get('extra_small_text'):
            score += 2
        if match_data.get('blind_recognition_feature'):
            score += 2
        if match_data.get('raised_paint'):
            score += 2
        if match_data.get('see_through'):
            score += 3
        if match_data.get('front_match'):
            score += 2
        if match_data.get('back_match'):
            score += 2

        print(f"{note_name.upper()} Score: {score}")

        if score > max_score:
            max_score = score
            identified_note = note_name

    # Define a threshold for considering a note as real (e.g., score >= 15)
    authenticity_threshold = 15

    if max_score >= authenticity_threshold and identified_note:
        return identified_note.upper()
    else:
        return "Unable to identify note or note is fake"

# ===========================
# Step 9: Main Function
# ===========================

def main():
    # Set display_steps to True to visualize preprocessing and detection steps
    display_steps = True
    
    # Load templates
    templates = load_templates()
    
    # Load and preprocess uploaded currency note images
    try:
        uploaded_note_images = load_uploaded_images(display_steps=display_steps)
    except IOError as e:
        print(e)
        sys.exit(1)
    
    # Align uploaded images with each template's front and back images
    # This step assumes that the uploaded note could be any of the currency notes
    # We'll align the uploaded images with each template and choose the best alignment
    aligned_uploaded_images = {}
    
    for note_name, note_templates in templates.items():
        print(f"\nAligning uploaded images with {note_name.upper()} templates...")
        
        front_template = note_templates.get('front', None)
        back_template = note_templates.get('back', None)
        
        if front_template is None or back_template is None:
            print(f"Warning: Front or back template missing for {note_name.upper()}. Skipping alignment.")
            aligned_uploaded_images[note_name] = (uploaded_note_images[0], uploaded_note_images[1])
            continue
        
        front_aligned = align_images(front_template, uploaded_note_images[0], display_steps=display_steps)
        back_aligned = align_images(back_template, uploaded_note_images[1], display_steps=display_steps)
        
        aligned_uploaded_images[note_name] = (front_aligned, back_aligned)
    
    # Detect features for each aligned template
    detection_results = {}
    
    for note_name, aligned_images in aligned_uploaded_images.items():
        detection_results[note_name] = detect_features(
            {note_name: templates[note_name]},
            aligned_images,
            display_steps=display_steps
        )[note_name]
    
    # Identify the currency note based on detection results
    final_result = identify_currency(detection_results)
    
    # Display final identification result
    print("\n=========================")
    print("     Detection Results    ")
    print("=========================")
    print(f"Identified Currency Note: {final_result}")

if _name_ == "_main_":
    main()