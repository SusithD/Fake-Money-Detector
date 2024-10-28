import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from dataset.dataset import CurrencyDataset  # Adjusted for the correct path

# Load the model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load your dataset
dataset = CurrencyDataset('dataset/annotations.json', 'dataset/images/')  # Adjusted path
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Fine-tuning logic
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Add an optimizer

for images, captions in dataloader:
    # Process images and captions
    pixel_values = processor(images, return_tensors="pt").pixel_values
    labels = tokenizer(captions, return_tensors="pt", padding=True, truncation=True).input_ids

    # Forward pass
    outputs = model(pixel_values=pixel_values, labels=labels)
    loss = outputs.loss

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f'Loss: {loss.item()}')

# Save the model
model.save_pretrained("fine_tuned_currency_model")
processor.save_pretrained("fine_tuned_currency_model")
tokenizer.save_pretrained("fine_tuned_currency_model")