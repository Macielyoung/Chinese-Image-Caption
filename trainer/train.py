from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from datasets import Dataset
import base64
from io import BytesIO
import pandas as pd
from PIL import Image
import numpy as np
import evaluate
import os
import argparse
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


ignore_pad_token_for_loss = True
metric = evaluate.load("rouge")

# text preprocessing step
def tokenization_fn(captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(captions, 
                       padding="max_length",
                       max_length=max_target_length).input_ids

    return labels


# image preprocessing step
def feature_extraction_fn(exist_image_contents):
    """
    Run feature extraction on images
    """
    encoder_inputs = feature_extractor(images=exist_image_contents, return_tensors="np")
    pixel_values = encoder_inputs.pixel_values
    
    return pixel_values


def preprocess_fn(examples, max_target_length, is_train=True):
    """Run tokenization + image feature extraction"""
    image_ids = examples['id']
    captions = examples['caption']
    
    exist_image_contents, exist_image_captions = [], []
    for image_id, image_caption in zip(image_ids, captions):
        try:
            if is_train:
                if image_id in muge_train_image_dict:
                    muge_train_image_content = muge_train_image_dict[image_id]
                    muge_train_img = Image.open(BytesIO(base64.urlsafe_b64decode(muge_train_image_content)))
                    muge_train_img = muge_train_img.convert("RGB")
                    exist_image_contents.append(muge_train_img)
                    exist_image_captions.append(image_caption)
            else:
                if image_id in muge_eval_image_dict:
                    muge_eval_image_content = muge_eval_image_dict[image_id]
                    muge_eval_img = Image.open(BytesIO(base64.urlsafe_b64decode(muge_eval_image_content)))
                    muge_eval_img = muge_eval_img.convert("RGB")
                    exist_image_contents.append(muge_eval_img)
                    exist_image_captions.append(image_caption)
        except Exception as e:
            print("image_id: {}, image_caption: {}, error: {}".format(image_id, image_caption, e), flush=True)
            continue
        
    assert len(exist_image_contents) == len(exist_image_captions)
         
    model_inputs = {}
    # This contains image path column
    model_inputs['labels'] = tokenization_fn(exist_image_captions, max_target_length)
    model_inputs['pixel_values'] = feature_extraction_fn(exist_image_contents)

    return model_inputs


def postprocess_text(preds, labels):
    # preds = [pred.strip() for pred in preds]
    # labels = [label.strip() for label in labels]

    # # rougeLSum expects newline after each sentence
    # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    # Rouge expects a blank between words for chinese
    preds = [" ".join(pred.strip()) for pred in preds]
    labels = [" ".join(label.strip()) for label in labels]
    
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)
    # print("decoded preds: {}, decoded_labels: {}".format(decoded_preds, decoded_labels))
    
    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            tokenizer=lambda x: x.split())
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def read_muge_caption_dataset(caption_file):
    muge_caption_df = pd.read_csv(caption_file)
    muge_caption_dataset = Dataset.from_pandas(muge_caption_df)
    return muge_caption_dataset


def read_muge_image_dataset(image_file):
    muge_image_df = pd.read_csv(image_file, header=None, names=['id', 'content'], sep='\t')
    muge_image_dict = muge_image_df.set_index(['id'])['content'].to_dict()
    return muge_image_dict


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, default="../datasets/muge/", help="data root")
    parser.add_argument("--image_root", type=str, default="../data/ECommerce-IC/", help="image content root")
    parser.add_argument("--image_pretrained", type=str, default="microsoft/beit-large-patch16-224-pt22k-ft22k", help="image pretrained model from huggingface")
    parser.add_argument("--text_pretrained", type=str, default="IDEA-CCNL/Wenzhong-GPT2-110M", help="text pretrained model from huggingface")
    parser.add_argument("--maxlength", type=int, default=128, help="max sentence length")
    parser.add_argument("--train_batch", type=int, default=32, help="train batch size")
    parser.add_argument("--eval_batch", type=int, default=64, help="eval batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="warmup step num")
    parser.add_argument("--eval_steps", type=int, default=1000, help="eval step num")
    parser.add_argument("--save_steps", type=int, default=1000, help="save step num")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="gradient accumulation step num")
    parser.add_argument("--weight decay", type=float, default=0.01, help="weight decay")
    
    parser.add_argument("--epoches", type=int, default=3, help="train epoch num")
    parser.add_argument("--save_num", type=int, default=3, help="save model num")
    parser.add_argument("--save_path", type=str, default="../models/image-caption-beit-large-wenzhong-gpt", help="image caption model save root")
    parser.add_argument("--task", type=str, default="image-caption", help="task name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # load muge caption dataset
    muge_train_caption_file = args.data_root + "train_muge_caption.csv"
    muge_train_caption_dataset = read_muge_caption_dataset(muge_train_caption_file)
    print(muge_train_caption_dataset)
    print("load muge train caption done", flush=True)

    muge_eval_caption_file = args.data_root + "valid_muge_caption.csv"
    muge_eval_caption_dataset = read_muge_caption_dataset(muge_eval_caption_file)
    print(muge_eval_caption_dataset)
    print("load muge eval caption done", flush=True)

    # load muge image content
    muge_train_image_file = args.image_root + "IC_train.tsv"
    muge_train_image_dict = read_muge_image_dataset(muge_train_image_file)
    print("load muge train image done", flush=True)

    muge_eval_image_file = args.image_root + "IC_valid.tsv"
    muge_eval_image_dict = read_muge_image_dataset(muge_eval_image_file)
    print("load muge eval image done", flush=True)

    # image encoder pretrained model and text decoder pretrained model
    # image pretrained model including ViT, BEiT, DeiT and Swin style model
    # text pretrained model including RoBERTa, GPT2, BERT and DistilBERT style model
    image_encoder_model = args.image_pretrained
    text_decoder_model = args.text_pretrained

    # image feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
    # text tokenizer
    tokenizer = AutoTokenizer.from_pretrained(text_decoder_model)
    # update model config
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = muge_train_caption_dataset.map(preprocess_fn,
                                                batched=True,
                                                fn_kwargs={"max_target_length": args.maxlength, "is_train": True},
                                                remove_columns=muge_train_caption_dataset.column_names)
    eval_dataset = muge_eval_caption_dataset.map(preprocess_fn,
                                                batched=True,
                                                fn_kwargs={"max_target_length": args.maxlength, "is_train": False},
                                                remove_columns=muge_eval_caption_dataset.column_names)
    print(train_dataset)
    print(eval_dataset)

    # load pretrained model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        image_encoder_model, text_decoder_model)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # making sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size
    # setting beam search parameter
    model.config.max_length = args.maxlength
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 5
    model.config.length_penalty = 2.0
    model.config.num_beams = 10
    model.decoder.resize_token_embeddings(len(tokenizer))

    # freezing the encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    tokenizer.save_pretrained(args.save_path)
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=args.save_num,
        load_best_model_at_end=True,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        output_dir=args.save_path,
        overwrite_output_dir=True,
        num_train_epochs=args.epoches,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        fp16=True,
        report_to="tensorboard"
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model()