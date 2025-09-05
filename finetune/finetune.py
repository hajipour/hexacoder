import os
import argparse
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset, load_from_disk, disable_caching
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_API_KEY"] = ""
seed = 42
wandb.login()
wandb.init(
    # set the wandb project where this run will be logged
    project="codegen-350m-8cwes-dataset_all_masked",

)

class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True

def tokenize_max(tokenizer, dataset):
  # Tokenize the dataset
  tokenized_dataset = tokenizer(dataset["text"])
  # Get the max length of the tokenized dataset
  max_length = max(map(len, tokenized_dataset["input_ids"]))
  print(f"Max length: {max_length}")

# Function to count trainable parameters
def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
if __name__ == "__main__":
  dataset = load_from_disk("datasets/") 
  model_name = "../models/codegen-2B-multi/"
  device_map = {"": 0}
  finetunes_model_name = "codegen-2B-8cwes-dataset_all_masked_params"
  model = AutoModelForCausalLM.from_pretrained(model_name, use_cache = False, device_map='auto')
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  # Shuffle the dataset["text"]
  dataset = dataset.shuffle(seed=42)
  # Devide the dataset into training and validation sets
  dataset = dataset.train_test_split(test_size=0.1, seed=seed)
  tokenize_max(tokenizer, dataset["train"])
  tokenize_max(tokenizer, dataset["test"])
  peft_config = LoraConfig(
      lora_alpha=16,
      lora_dropout=0.1,
      r=8,
      bias="none",
      task_type="CAUSAL_LM",
      target_modules = ["qkv_proj"]
  )

  trainingArgs = TrainingArguments(
    output_dir=finetunes_model_name,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1, # 16:1, 4:2
    gradient_checkpointing=False,
    evaluation_strategy="epoch",
    optim="paged_adamw_32bit",
    do_eval = True,
    # eval_steps=74, # BS = 16 -> 74, BS = 4 -> 148
    # logging_steps=74,
    logging_strategy="epoch",
    save_strategy="epoch",
    # save_steps=74,
    load_best_model_at_end= True,
    learning_rate=5e-4,
    weight_decay=0.001,
    # max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=False, # For a better training performance set it to True
    lr_scheduler_type="cosine",
    report_to="wandb",
    ddp_find_unused_parameters= False,
    seed=seed
  )
  
  trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length=512,
    tokenizer=tokenizer,
    dataset_text_field="text",
    # neftune_noise_alpha=5,
    args=trainingArgs
  )
  count_trainable_parameters(model)
  # trainer.add_callback(EvaluateFirstStepCallback())
  # trainer.train() 
  wandb.finish()
