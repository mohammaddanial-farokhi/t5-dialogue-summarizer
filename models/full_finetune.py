from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from data.load_dataset import load_model_dataset
from models.base_model import load_base_model
from data.preprocess import prepare_tokenized_dataset
import os
import time
from pathlib import Path
import shutil

CURRENT_DIR = Path(__file__).parent
MODEL_CACHE_DIR = CURRENT_DIR / "cached_finetuned_model"

def get_or_train_model(force_retrain=False):

    if not force_retrain and MODEL_CACHE_DIR.exists():
        print(f"Loading fine-tuned model from {MODEL_CACHE_DIR} ...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CACHE_DIR)
        _, tokenizer = load_base_model()  
        return model, tokenizer

    
    print("Preparing tokenized dataset...")
    tokenized_datasets, tokenizer = prepare_tokenized_dataset(force_recompute=force_retrain)
    
    
    print("Loading base model...")
    model, _ = load_base_model()  

    
    output_dir = f"./training_output_{int(time.time())}"  
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-5,
        num_train_epochs=1,          
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        logging_steps=10,
        report_to="none",
        save_strategy="epoch",       
        evaluation_strategy="epoch", 
        predict_with_generate=True,
        fp16=True,                   
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    
    print("Starting training...")
    trainer.train()

    
    print(f"Saving fine-tuned model to {MODEL_CACHE_DIR} ...")
    trainer.save_model(MODEL_CACHE_DIR)
    tokenizer.save_pretrained(MODEL_CACHE_DIR)  

    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print("Model saved successfully.")
    return model, tokenizer