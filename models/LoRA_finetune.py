from peft import LoraConfig, get_peft_model, TaskType,PeftModel
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from models.base_model import load_base_model
import time
from data.preprocess import prepare_tokenized_dataset
import os
from pathlib import Path
import torch


CURRENT_DIR = Path(__file__).parent
MODEL_CACHE_DIR = CURRENT_DIR / "cached_peft_model"

def get_train_lora(force_retrain=False):
    if not force_retrain and MODEL_CACHE_DIR.exists():
        print(f"Loading LoRA model from {MODEL_CACHE_DIR} ...")
        
        # بارگذاری مدل پایه (همان مدلی که LoRA روی آن آموزش دیده)
        base_model_name = "google/flan-t5-base"
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, MODEL_CACHE_DIR)
        return model, tokenizer
    
    print("Training new LoRA model...")

    base_model, base_tokenizer = load_base_model()
    tokenized_datasets, _ = prepare_tokenized_dataset()

    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
    )


    peft_model = get_peft_model(base_model, lora_config)
    # print(print_number_of_trainable_model_parameters(peft_model))

    output_dir = f"./peft-dialogue-summary-training-{str(int(time.time()))}"

    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-3,
        num_train_epochs=1,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
        label_names=["labels"],
    )

    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=tokenized_datasets["train"],
    )
    peft_trainer.train()

    peft_model_path = "./cached_peft_model"

    peft_trainer.model.save_pretrained(peft_model_path)
    base_tokenizer.save_pretrained(peft_model_path)

    return peft_model, base_tokenizer
