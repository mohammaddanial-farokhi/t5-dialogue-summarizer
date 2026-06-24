from data.load_dataset import load_model_dataset
from models.base_model import load_base_model, count_trainable
from inference.incontext_zeroshot import create_ZS_prompt, zero_shot
from inference.incontext_fewshots import create_FS_prompt, few_shots
from data.preprocess import prepare_tokenized_dataset
from models.full_finetune import get_or_train_model
from utils.sample_check import one_sample
from evaluation.rouge_metric import rouge_calculation
from models.LoRA_finetune import get_train_lora
from utils.make_report import reporting

###-_-_-_-_-_-_-_-##
# 1- loading model
###-_-_-_-_-_-_-_-##
# main_dataset = load_model_dataset()
# base_model, base_tokenizer = load_base_model()
# print(count_trainable(model))

###-_-_-_-_-_-_-_-##
# 2- incontext learing with base model
###-_-_-_-_-_-_-_-##
# prompt, ZS_dial, ZS_summ = create_ZS_prompt()
# # print(prompt)
# zero_shot_ans = zero_shot(model, tokenizer, prompt, ZS_dial, ZS_summ)
# print(zero_shot_ans)

# prompt, FS_dial, FS_summ = create_FS_prompt()
# # print(prompt)
# few_shots_ans = few_shots(model, tokenizer, prompt, FS_dial, FS_summ)
# print(few_shots_ans)

###-_-_-_-_-_-_-_-##
# 3- pre process dataset and tokenized prompts
###-_-_-_-_-_-_-_-##
# tokenized_datasets, predata_tokenizer = prepare_tokenized_dataset()

###-_-_-_-_-_-_-_-##
# 4-full-fine tune modeling
###-_-_-_-_-_-_-_-##

# full_fine_model, full_fine_tokenizer = get_or_train_model()

# result = one_sample(full_fine_model, full_fine_tokenizer, tokenized_datasets)
# print(result)

###-_-_-_-_-_-_-_-##
# 5- full-fine tune evaluation
###-_-_-_-_-_-_-_-##
# full_fine_rouge = rouge_calculation(get_or_train_model)


###-_-_-_-_-_-_-_-##
# 6-LoRA tune modeling
###-_-_-_-_-_-_-_-##
# lora_model, lora_tokenizer = get_train_lora()

# result = one_sample(lora_model, lora_tokenizer, tokenized_datasets)
# print(result)

# LoRA_rouge = rouge_calculation(get_train_lora)

###-_-_-_-_-_-_-_-##
# 7-final report
###-_-_-_-_-_-_-_-##
# reporting()