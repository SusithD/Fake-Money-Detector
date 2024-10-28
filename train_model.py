import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AdamW
from torch.utils.data import DataLoader
from datasets import load_from_disk

# Hyperparameters
batch_size = 8
num_epochs = 5
learning_rate = 5e-5

# Load the processed dataset
def load_dataset():
    return load_from_disk("processed_dataset")

# Collate function for DataLoader to stack tensors
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])  # Stack pixel values
    labels = torch.stack([item["labels"] for item in batch])  # Stack labels
    return {"pixel_values": pixel_values, "labels": labels}

# Create DataLoader
def create_dataloader(dataset):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Define the training function
def train(model, dataloader, optimizer, device):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Main function
def main():
    dataset = load_dataset()
    dataloader = create_dataloader(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    train(model, dataloader, optimizer, device)

if __name__ == "__main__":
    main()
