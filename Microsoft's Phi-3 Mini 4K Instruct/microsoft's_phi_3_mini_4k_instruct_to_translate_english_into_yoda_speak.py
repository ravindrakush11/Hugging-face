# -*- coding: utf-8 -*-
"""Microsoft's Phi-3 Mini 4K Instruct to translate English into Yoda-speak.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1R_ldZhT-I85YbUrOSdJUcaaSUMVhsHc8

# Microsoft's Phi-3 Mini 4K Instruct, to translate English into Yoda-speak
"""

!pip install -q transformers==4.46.2 peft==0.13.2 accelerate==1.1.1 trl==0.12.1 bitsandbytes==0.45.2 datasets==3.1.0 huggingface-hub==0.26.2 safetensors==0.4.5 pandas==2.2.2 matplotlib==3.8.0 numpy==1.26.4

# !pip install -q datasets bitsandbytes trl

# Imports
import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# Loading a Quantized Base Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32,
)
repo_id = 'microsoft/Phi-3-mini-4k-instruct'
model = AutoModelForCausalLM.from_pretrained(
    repo_id, device_map="cuda:0", quantization_config=bnb_config
)

print(model.get_memory_footprint()/1e6)

model

# Setting Up Low-Rank Adapters(LoRA)
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r = 8,
    lora_alpha = 16,
    bias = "none",
    lora_dropout = 0.05,
    task_type = "CAUSAL_LM",
    target_modules = ['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
)
model = get_peft_model(model, config)
model

print(model.get_memory_footprint()/1e6)

trainable_parms, tot_parms = model.get_nb_trainable_parameters()
print(f"Trainable parameters:  {trainable_parms/1e6:.2f}M")
print(f'Total parameters: {tot_parms/1e6:.2f}M')
print(f'Fraction of trainable parameters: {100*trainable_parms/tot_parms:.2f}%')

"""Now the model is read for **FineTuning**

## Formatting Dataset
"""

# Formatting Dataset
dataset = load_dataset('dvgodoy/yoda_sentences', split="train")
dataset

dataset[0]

# Renaming and Removing Columns
dataset = dataset.rename_column("sentence", "prompt")
dataset = dataset.rename_column("translation_extra", "completion")
dataset = dataset.remove_columns(["translation"])
dataset

# dataset[0][prompt]

messages = [
    {"role": "user", "content": dataset[0]['prompt']},
    {"role": "assistant", "content": dataset[0]['completion']}
]
messages

# # # convert the dataset to the conversational format using the format_dataset() function
# def format_dataset(examples):
#   if isinstance(examples["prompt"], list):
#     output_texts = []
#     for i in range(len(examples["prompt"])):
#       converted_sample = [
#           {"role": "user", "content": examples["prompt"][i]},
#           {"role": "assistant", "content": examples["completion"][i]},
#       ]
#       output_texts.append(converted_sample)
#     return {'messages': output_texts}
#   else:
#     converted_sample = [
#         {"role": "user", "content": examples["prompt"]},
#         {"role": "assistant", "content": examples["completion"]},

#     ]
#     return {"messages": converted_sample}

def format_dataset(examples):
    if isinstance(examples["prompt"], list):
        output_texts = []
        for i in range(len(examples["prompt"])):
            converted_sample = [
                {"role": "user", "content": examples["prompt"][i]},
                {"role": "assistant", "content": examples["completion"][i]},
            ]
            output_texts.append(converted_sample)
        return {'messages': output_texts}
    else:
        converted_sample = [
            {"role": "user", "content": examples["prompt"]},
            {"role": "assistant", "content": examples["completion"]},
        ]
        return {'messages': converted_sample}

dataset = dataset.map(format_dataset).remove_columns(['prompt', 'completion'])

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)
tokenizer.chat_template



print(tokenizer.apply_chat_template(messages, tokenize=False))

tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id



"""## Fine-Tuning with SFTTrainer"""

sft_config = SFTConfig(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs = {'use_reentrant': False},
    gradient_accumulation_steps=1,
    per_device_train_batch_size=16,
    auto_find_batch_size=True,

    max_seq_length=64,
    packing=True,

    num_train_epochs = 10,
    learning_rate = 3e-4,

    optim = 'paged_adamw_8bit',

    logging_steps=10,
    logging_dir = './logs',
    output_dir = './phi3-mini-yoda-adapter',
    report_to = 'none'
)

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    args = sft_config,
    train_dataset = dataset,
)

dl = trainer.get_train_dataloader()
batch = next(iter(dl))

batch['input_ids'][0], batch['labels'][0]

trainer.train()

"""## Querying the Model"""

def gen_prompt(tokenizer, sentence):
  converted_sample = [
      {"role": "user", "content": sentence},
  ]
  prompt = tokenizer.apply_chat_template(converted_sample,
                                         tokenize = False,
                                         add_generation_prompt = True)
  return prompt

# Genrating a prompt for an example sentence
sentence = "The Force is strong in you!"
prompt = gen_prompt(tokenizer, sentence)
print(prompt)

def generate(model, tokenizer, prompt, max_new_tokens=64, skip_special_tokens=False):
    tokenized_input = tokenizer(prompt, add_special_tokens = False, return_tensors="pt").to(model.device)

    model.eval()
    generation_output = model.generate(**tokenized_input,
                                      eos_token_id = tokenizer.eos_token_id,
                                      max_new_tokens = max_new_tokens)

    output = tokenizer.batch_decode(generation_output,
                                    skip_special_tokens = skip_special_tokens)
    return output[0]

print(generate(model, tokenizer, prompt))

# Saving the Adapter
trainer.save_model('local-phi3-mini-yoda-adapter')

os.listdir('local-phi3-mini-yoda-adapter')

"""### Sharing of the adapter on HF Hub"""

from huggingface_hub import login
login()

trainer.push_to_hub()