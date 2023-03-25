import json
import pandas as pd
from transformers import AutoFeatureExtractor
import base64
from io import BytesIO
from PIL import Image

MAX_IMAGE_SIZE = 800000

def load_caption(caption_file, feature_extractor, image_content_dict):
    max_caption_length = 0
    captions = []
    with open(caption_file, 'r') as f:
        for lid, line in enumerate(f):
            try:
                caption = json.loads(line)
                image_id = caption['image_id']
                
                image_content = image_content_dict.get(image_id, "")
                if image_content == "":
                    continue
                img = Image.open(BytesIO(base64.urlsafe_b64decode(image_content)))
                img = img.convert("RGB")
                if img.mode != "RGB":
                    continue
                img_size = img.width * img.height
                if img_size > MAX_IMAGE_SIZE:
                    continue
                encoder_inputs = feature_extractor(images=img, return_tensors="np")
                image_texts = caption['text']
                for text in image_texts:
                    text_length = len(text)
                    if text_length > max_caption_length:
                        max_caption_length = text_length
                    item = {'id': image_id,
                            'caption': text}
                    captions.append(item)
                if lid % 1000 == 0:
                    print("caption_file: {}, proceed line {}".format(caption_file, lid), flush=True)
            except Exception as e:
                print("image_id: {}, caption: {}, error: {}".format(image_id, caption, e), flush=True)
                continue
    return captions, max_caption_length


# load muge image content
muge_train_image_file = "../data/ECommerce-IC/IC_train.tsv"
muge_train_image_df = pd.read_csv(muge_train_image_file, header=None, names=['id', 'content'], sep='\t')
muge_train_image_dict = muge_train_image_df.set_index(['id'])['content'].to_dict()
print("load muge train image done", flush=True)

muge_eval_image_file = "../data/ECommerce-IC/IC_valid.tsv"
muge_eval_image_df = pd.read_csv(muge_eval_image_file, header=None, names=['id', 'content'], sep='\t')
muge_eval_image_dict = muge_eval_image_df.set_index(['id'])['content'].to_dict()
print("load muge eval image done", flush=True)

image_encoder_model = "microsoft/beit-large-patch16-224-pt22k-ft22k"
# image feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)

train_caption_file = "../data/ECommerce-IC/IC_train.jsonl"
train_captions, train_caption_max_len = load_caption(train_caption_file, feature_extractor, muge_train_image_dict)
train_caption_df = pd.DataFrame(train_captions)
print(train_caption_df.shape, train_caption_max_len)
train_file = "../datasets/muge/train_muge_caption.csv"
train_caption_df.to_csv(train_file)

eval_caption_file = "../data/ECommerce-IC/IC_valid.jsonl"
eval_captions, eval_caption_max_len = load_caption(eval_caption_file, feature_extractor, muge_eval_image_dict)
eval_caption_df = pd.DataFrame(eval_captions)
print(eval_caption_df.shape, eval_caption_max_len)
eval_file = "../datasets/muge/valid_muge_caption.csv"
eval_caption_df.to_csv(eval_file)