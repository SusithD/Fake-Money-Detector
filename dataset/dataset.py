import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CurrencyDataset(Dataset):
    def __init__(self, annotation_file, images_dir, transform=None):
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)["annotations"]
        
        self.images_dir = images_dir

        # Define the transformations, including resizing
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the desired dimensions
            transforms.ToTensor(),           # Convert image to tensor
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_id = self.annotations[idx]["image_id"]
        caption = self.annotations[idx]["caption"]
        
        # Load image
        img_path = os.path.join(self.images_dir, image_id)
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, caption