import os
import json
from PIL import Image
from torch.utils.data import Dataset

class CurrencyDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)['annotations']
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.annotations[idx]['image_id'])
        image = Image.open(img_name)
        caption = self.annotations[idx]['caption']
        
        if self.transform:
            image = self.transform(image)

        return image, caption

# Example usage
if __name__ == "__main__":
    dataset = CurrencyDataset('dataset/annotations.json', 'images/')
    print(f'Total samples: {len(dataset)}')
