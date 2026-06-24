import evaluate
import torch
import random
from models.base_model import load_base_model
from data.load_dataset import load_model_dataset
import numpy as np
import evaluate

dataset = load_model_dataset()


def rouge_calculation(target_model_loading, target_number=10):
    base_model, base_tokenizer = load_base_model()
    target_model, target_tokenizer = target_model_loading()

    main_index = random.sample(range(len(dataset["test"])), target_number)
    main_refrence = [(dataset["test"][idx]["summary"]) for idx in main_index]

    def create_prompt(dial):
        prompt = f""" Summarize the following conversation. Try to reason on the conversation and then reply the summary.
            {dial}
            Summary:
            """
        return prompt

    main_prompts = [create_prompt(dataset["test"][idx]["dialogue"]) for idx in main_index]

    def model_text_out(prompts, model, tokenizer):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        outputs = model.generate(**inputs, max_new_tokens=100, num_beams=4, early_stopping=True)

        results = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

        if results:
            print("✅ First prediction sample:", results[0])
        else:
            print("⚠️ No predictions generated!")
        # ========================
        return results

    base_predictions = model_text_out(main_prompts, base_model, base_tokenizer)
    target_predictions = model_text_out(main_prompts, target_model, target_tokenizer)

    print(f"base_predictions count: {len(base_predictions)}")
    print(f"target_predictions count: {len(target_predictions)}")

    def compute_rouge(predictions, references, use_stemmer=True):

        print(f"📊 len(predictions): {len(predictions)}, len(references): {len(references)}")

        rouge = evaluate.load("rouge")

        results = rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=True,
            use_stemmer=use_stemmer,
        )
        print("🔍 RAW ROUGE OUTPUT:", results)
        return  {k: v for k, v in results.items() if k.startswith("rouge")}

    def compare_models(base_summaries, target_summaries, references, target_name="TARGET"):

        base_results = compute_rouge(base_summaries, references)
        target_results = compute_rouge(target_summaries, references)

        print("=" * 50)
        print("BASE MODEL RESULTS:")
        for k, v in base_results.items():
            print(f"{k}: {v:.4f}")

        print("\n" + "=" * 50)
        print(f"{target_name.upper()} MODEL RESULTS:")
        for k, v in target_results.items():
            print(f"{k}: {v:.4f}")

        print("\n" + "=" * 50)
        print(f"IMPROVEMENT OF {target_name.upper()} OVER BASE MODEL:")
        for key in base_results.keys():
            base_val = base_results[key]
            target_val = target_results[key]
            improvement_percent = ((target_val - base_val) / base_val) * 100 if base_val != 0 else 0
            print(f"{key}: {improvement_percent:.2f}%")

        return base_results, target_results

    base_results, target_results = compare_models(
        base_summaries=base_predictions,
        target_summaries=target_predictions,
        references=main_refrence,
        target_name="full-finetune",
    )

    return base_results, target_results


# device = "cuda" if torch.cuda.is_available() else "cpu"
# original_model = base_model.to(device)
# instruct_model = instruct_model.to(device)
