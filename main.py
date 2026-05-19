import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("image.jpg")).unsqueeze(0)

labels = ["a dog", "a cat", "a car", "a person"]
text = tokenizer(labels)

with torch.no_grad():
    probs = (model.encode_image(image) @ model.encode_text(text).T).softmax(dim=-1)[0]

for label, prob in zip(labels, probs):
    print(f"{label}: {prob:.1%}")