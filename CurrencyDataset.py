import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CurrencyDataset(Dataset):
    def __init__(self, image_dir, annotations_file, transform=None, tokenizer=None):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        # Extract image ids and captions
        self.images = [os.path.join(image_dir, ann['image_id']) for ann in self.annotations]
        self.captions = [ann['caption'] for ann in self.annotations]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image path and caption
        img_name = self.images[idx]
        caption = self.captions[idx]

        # Load the image
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Debugging prints
        print(f"Image: {img_name}, Caption: {caption}")

        # Tokenize the caption
        tokenized_caption = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=50)

        return {
            "pixel_values": image,
            "labels": tokenized_caption['input_ids'].squeeze(0)  # Ensure proper tensor shape
        }
