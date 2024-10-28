from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

# Load model, processor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("./currency_caption_model")
processor = ViTImageProcessor.from_pretrained("./currency_caption_model")
tokenizer = AutoTokenizer.from_pretrained("./currency_caption_model")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    sample_image_path = "images/lkr20_front.jpg"
    caption = generate_caption(sample_image_path)
    print("Generated Caption:", caption)
