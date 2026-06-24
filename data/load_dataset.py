from datasets import load_dataset

def load_model_dataset(name="knkarthick/dialogsum"):
    dataset = load_dataset(name)
    # print(dataset)
    return dataset
    
