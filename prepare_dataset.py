import json
import torch
import os
from PIL import Image
from transformers import ViTImageProcessor, AutoTokenizer
from datasets import Dataset

# Initialize processor and tokenizer
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load dataset
def load_data():
    with open("dataset/annotations.json", "r") as f:
        data = json.load(f)
    if "annotations" in data:
        return data["annotations"]
    else:
        raise ValueError("Key 'annotations' not found in JSON")

# Preprocess function
def preprocess(example):
    image_path = os.path.join("dataset", "images", example["image_id"])
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    # Process the image and remove the extra batch dimension
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

    if pixel_values is None or pixel_values.size(0) == 0:
        print(f"Warning: pixel_values is None or empty for image: {example['image_id']}")
        return None

    # Tokenize the caption
    inputs = tokenizer(example["caption"], padding="max_length", max_length=128, truncation=True)

    return {
        "pixel_values": pixel_values,
        "labels": torch.tensor(inputs["input_ids"], dtype=torch.long)
    }

# Process and save the dataset
if __name__ == "__main__":
    data = load_data()
    dataset = Dataset.from_list(data)
    processed_dataset = dataset.map(lambda example: preprocess(example), remove_columns=dataset.column_names)
    processed_dataset = processed_dataset.filter(lambda example: example is not None)
    processed_dataset.save_to_disk("processed_dataset")
