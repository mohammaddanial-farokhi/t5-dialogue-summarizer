from data.load_dataset import load_model_dataset
import random
from data.preprocess import prepare_tokenized_dataset
import torch
from models.base_model import load_base_model
from models.full_finetune import get_or_train_model
from models.LoRA_finetune import get_train_lora
import pandas as pd
from datetime import datetime
import shutil
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dataset = load_model_dataset()
tokenized_dataset, _ = prepare_tokenized_dataset()  

base_model,base_tokenizer=load_base_model()
base_model = base_model.to(device) 

full_model,full_tokenizer=get_or_train_model()
full_model = full_model.to(device)

lora_model,lora_tokenizer=get_train_lora()
lora_model = lora_model.to(device)

def reporting(num_samples=50, output_path="final_results.csv"):
   
    index = []
    for i in range(num_samples):
        rand_int = random.randint(0, len(dataset["test"]) - 1)
        index.append(rand_int)

    results = []

    def generate_summary(model, tokenizer, input_ids):
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device) 
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_tensor,
                max_new_tokens=100,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    for idx in index:
        dial=dataset["test"][idx]["dialogue"]
        hum_s=dataset["test"][idx]["summary"]
        input_ids = tokenized_dataset["test"][idx]["input_ids"]  

        base_summary = generate_summary(base_model, base_tokenizer, input_ids)
        full_summary = generate_summary(full_model, full_tokenizer, input_ids)
        lora_summary = generate_summary(lora_model, lora_tokenizer, input_ids)

        results.append({
            "dialogue": dial,
            "human_summary": hum_s,
            "base_model_summary": base_summary,
            "full_finetune_summary": full_summary,
            "lora_summary": lora_summary,
        })


    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_file = f"final_results_{timestamp}.csv"
    df.to_csv(new_file, index=False, encoding="utf-8-sig")
    shutil.copy(new_file, "final_results_latest.csv")
    print(f"New report saved: {new_file}")
    print(f"Latest report updated: final_results_latest.csv")

    return df
