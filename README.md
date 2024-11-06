
---

# Currency Note Recognition System

An AI-powered Currency Note Recognition System, developed specifically for Sri Lankan currency, designed to authenticate and educate users about unique security and cultural features. This project combines state-of-the-art image processing and deep learning techniques to provide a reliable and user-friendly solution for counterfeit detection and currency education.
![Screenshot 2024-11-05 173706](https://github.com/user-attachments/assets/8d2e6543-69c2-4b94-94d2-563a3cc1a2cd)
![Screenshot 2024-11-05 173654](https://github.com/user-attachments/assets/684bc66a-5c2f-4f2a-8002-ba390ff9ff81)
![Screenshot 2024-11-05 175301](https://github.com/user-attachments/assets/8f3634f1-2a9c-4c5f-a46b-c116f953aa0a)
![Screenshot 2024-11-05 175313](https://github.com/user-attachments/assets/682e8c09-ed4c-4c68-9e44-45213f2c8333)
![Screenshot 2024-11-05 175328](https://github.com/user-attachments/assets/dd92e34c-c76f-4cfc-8575-dbe7aafc2a5d)
![Screenshot 2024-11-05 175335](https://github.com/user-attachments/assets/085f519c-cf2f-4887-b0b5-096a3b9da2e0)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technical Components](#technical-components)
4. [Model Training and Fine-Tuning](#model-training-and-fine-tuning)
5. [Installation and Setup](#installation-and-setup)
6. [Usage](#usage)
7. [Data and Preprocessing](#data-and-preprocessing)
8. [Future Work](#future-work)
9. [Acknowledgments](#acknowledgments)

---

## Project Overview

The Currency Note Recognition System leverages AI and deep learning for secure currency authentication and detailed feature descriptions. This system was tailored to Sri Lankan currency, utilizing specific image processing techniques to recognize visual symbols, security elements, and cultural markers on each note. The project aims to educate users on identifying currency-specific features while providing an accurate, robust solution to counterfeit detection.

---

## Features

- **Real-Time Currency Authentication**: Scans and verifies the authenticity of Sri Lankan currency notes.
- **Security Feature Detection**: Uses multiple image processing methods to detect watermarks, UV symbols, micro-text, and other security markers.
- **Educational Descriptions**: Generates detailed explanations of currency features, fostering public knowledge of unique cultural symbols and security measures.
- **Interactive Interface**: User-friendly interface providing insights into both front and back note designs.
- **Object Detection and Classification**: YOLO model trained to identify symbols unique to Sri Lankan currency, like butterflies, birds, traditional dancers, and landmarks.

---

## Technical Components

### 1. Image Preprocessing

To prepare currency images for high-accuracy feature extraction, the following image processing techniques were used:

- **Color Conversion**: Adjusting color spaces for optimal detection.
- **Noise Reduction with Bilateral Filtering**: Reducing image noise while preserving edges.
- **Adaptive Thresholding**: Enhancing contrast for clearer feature extraction.
- **Morphological Operations**: Refining image shapes for better analysis.
- **Edge Detection (Canny)**: Highlighting edges to assist in shape recognition.
- **Template Matching and Feature Matching (SIFT, ORB)**: For verifying pattern consistency and template alignment.
- **Dominant Color Detection (KMeans)**: Identifying color schemes to aid classification.
- **YOLO Object Detection**: Detecting symbols like butterflies, birds, dancers, and landmarks.
- **OCR (Optical Character Recognition)**: Extracting textual information like note value, denomination, and unique identifiers.

### 2. Security Feature Detection

The system employs specific techniques to analyze and confirm the presence of essential security features:

- **Watermark Detection**: Ensures the authenticity of embedded watermarks.
- **UV Symbol Detection**: Validates patterns visible only under ultraviolet light.
- **Micro-Text Analysis**: Detects small inscriptions challenging to replicate, adding a level of security.
- **Pattern Recognition**: Examines see-through registers, colors, and layouts that confirm alignment with authentic currency templates.

### 3. Model Training and Educational Captioning

The system was fine-tuned using Hugging Face’s `nlpconnect/vit-gpt2-image-captioning` model for currency-specific description generation. Key components include:

- **Detailed Descriptions**: Currency-specific descriptions of note symbols, colors, and cultural elements.
- **Intuitive User Guidance**: Generated captions help users understand security markers, symbols, and notable features, bridging the gap between currency recognition and education.

### 4. Custom Object Detection with YOLO

A custom-trained YOLO model accurately detects specific visual elements, crucial for authenticating Sri Lankan currency:

- **Fine-Tuning for Sri Lankan Symbols**: Custom training dataset with labeled images of butterflies, birds, dancers, and landmarks unique to Sri Lanka.
- **Multi-Stage Detection Pipeline**: Combines YOLO with SIFT and ORB for accurate symbol detection and classification.

---

## Model Training and Fine-Tuning

1. **Dataset Creation**: Curated images of Sri Lankan currency notes, labeled with annotations detailing the front and back designs, colors, symbols, and security features.
2. **Preprocessing and Augmentation**: Applied techniques like rotation, scaling, and color adjustments to diversify the dataset and enhance model robustness.
3. **Fine-Tuning Hugging Face VisionEncoderDecoderModel**: Adapted for currency-specific captions, refining output to include note-related vocabulary and avoiding general terms.
4. **YOLO Training**: YOLO model was fine-tuned to detect specific Sri Lankan currency symbols, using a labeled dataset of currency symbols and landmarks.
5. **Model Evaluation**: Evaluated for accuracy in detection, description quality, and efficiency in real-time usage.

---

## Installation and Setup

### Requirements

- Python 3.8+
- Flask
- Hugging Face Transformers
- OpenCV
- PyTorch
- YOLOv5 (or any suitable version)

### Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/currency-note-recognition.git
   cd currency-note-recognition
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Prepare Dataset**:
   - Place the dataset in the `dataset` directory as structured in the repository.

4. **Run the Application**:
   ```bash
   python app.py
   ```

---

## Usage

1. **Upload Currency Image**: Start the application and upload an image of the currency note.
2. **View Authentication Results**: The application will display authenticity results based on detected security features.
3. **Educational Insights**: See generated captions detailing unique currency features and cultural elements.

---

## Data and Preprocessing

The project’s dataset includes high-resolution images of Sri Lankan currency notes with labeled annotations. Preprocessing steps include color correction, noise reduction, and augmentation to improve model training and accuracy.

### Data Structure

The dataset should be organized in the following structure:
```
dataset/
│
├── images/
│   ├── <currency_note_image1>.jpg
│   ├── <currency_note_image2>.jpg
│
└── annotations.json
```

Annotations should detail each note's primary colors, symbols, and security features, allowing the model to generate accurate descriptions.

---

## Future Work

1. **Expand Currency Database**: Incorporate more currency types for broader coverage.
2. **Real-Time Detection**: Implement to make the app faster and more responsive.
3. **Enhanced User Interface**: Develop a more interactive and intuitive interface.

---

## Acknowledgments

This project was inspired by the need for accessible, AI-driven solutions to counterfeit currency detection, emphasizing Sri Lanka’s unique currency features. Special thanks to [Hugging Face](https://huggingface.co/) for providing model support and the open-source community for their valuable tools.

---
