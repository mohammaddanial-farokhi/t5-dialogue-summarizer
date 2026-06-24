from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from data.load_dataset import load_model_dataset

dataset = load_model_dataset()


def load_base_model(model_name="google/flan-t5-base"):
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return original_model, tokenizer


def count_trainable(model):
    all_params = model.num_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent = 100 * trainable_params / all_params if all_params > 0 else 0
    return f"Trainable: {trainable_params}\nAll: {all_params}\nPercent: {percent:.2f}%"

