import random
import torch
from data.load_dataset import load_model_dataset

dataset = load_model_dataset()

def one_sample(model, tokenizer, tokenized_datasets, sample_index=None):

    if sample_index is None:
        sample_index = random.randint(0, len(dataset["test"]) - 1)
    
    main_dial = dataset["test"][sample_index]["dialogue"]
    main_summ = dataset["test"][sample_index]["summary"]
    
    sample_input_ids = tokenized_datasets["test"][sample_index]["input_ids"]
    
    input_tensor = torch.tensor(sample_input_ids).unsqueeze(0)
    
    with torch.no_grad():  
        output_ids = model.generate(
            input_ids=input_tensor,
            max_new_tokens=100,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,  
        )
    
    generated_summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    dash = "=" * 60
    result = (
        f"\n{dash}\n"
        f"MAIN DIALOGUE :\n{dash}\n{main_dial}\n\n"
        f"{dash}\nHUMAN SUMMARY :\n{dash}\n{main_summ}\n\n"
        f"{dash}\nMODEL SUMMARY :\n{dash}\n{generated_summary}\n"
    )
    
    return result