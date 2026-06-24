from data.load_dataset import load_model_dataset
from models.base_model import load_base_model

dataset = load_model_dataset()
model, tokenizer = load_base_model()


def get_short_examples(dataset, num_examples=2, max_tokens=250):
    short_indices = []
    for i in range(len(dataset["test"])):
        if len(dataset["test"][i]["dialogue"]) < max_tokens:
            short_indices.append(i)
        if len(short_indices) >= num_examples:
            break
    return short_indices[:num_examples]


def create_FS_prompt():
    indexs = get_short_examples(dataset)
    dial = list(dataset["test"][i]["dialogue"] for i in indexs)
    summ = list(dataset["test"][i]["summary"] for i in indexs)

    main_index = 59
    main_dial = dataset["test"][main_index]["dialogue"]
    main_summ = dataset["test"][main_index]["summary"]

    prompt = f""" You are an expert at summarizing dialogues.Your task is to write a concise summary that captures the key points of the conversation between #Person1# and #Person2#.Follow the style of the examples below.
                  Example 1: Dialogue:{dial[0]}, Summary: {summ[0]}
                  Example 2: Dialogue:{dial[1]}, Summary: {summ[1]}

                  Now, summarize the following dialogue in the same style:
                  Dialogue:{main_dial}
                  Summary:
            """
    return prompt, main_dial, main_summ


def few_shots(model, tokenizer, prompt, dial, summ):
    input = tokenizer(prompt, return_tensors="pt")
    model_ans = model.generate(input["input_ids"], max_new_tokens=200)[0]
    output = tokenizer.decode(model_ans, skip_special_tokens=True)

    return f"input:\n {dial}\nhuman summary:{summ}\nmodel FS summ:\n{output}"
