
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import os
import readline


pretrained = "../models/checkpoint-5000"
model = VisionEncoderDecoderModel.from_pretrained(pretrained)
feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)
print("load model, feature extractor and tokenizer done!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 64
num_beams = 10
gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": True}
def predict_step(image_path):
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert("RGB")

    pixel_values = feature_extractor(images=i_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


if __name__ == "__main__":
    while True:
        image_path = input("please input your image path:\n")
        if os.path.exists(image_path):
            preds = predict_step(image_path)
            print(preds)
        else:
            print("please input correct image path!")
