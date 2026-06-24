from data.load_dataset import load_model_dataset
from models.base_model import load_base_model

dataset = load_model_dataset()
model, tokenizer = load_base_model()


def create_ZS_prompt(index=59):
    dial = dataset["test"][index]["dialogue"]
    summ = dataset["test"][index]["summary"]

    prompt = f""" Summarize the following conversation. Try to reason on the conversation and then reply the summary.
            {dial}
            Summary:
            """
    return prompt, dial, summ


def zero_shot(model, tokenizer, prompt, dial, summ):
    input = tokenizer(prompt, return_tensors="pt")
    model_ans = model.generate(input["input_ids"], max_new_tokens=200)[0]
    output = tokenizer.decode(model_ans, skip_special_tokens=True)

    return f"input:\n {dial}\nhuman summary:{summ}\nmodel ZS summ:\n{output}"
