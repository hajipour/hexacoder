import os
import argparse
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling
from datasets import Dataset, load_from_disk, disable_caching
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from accelerate import Accelerator
import wandb

torch.cuda.empty_cache()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_API_KEY"] = ""
seed = 42
wandb.login()
wandb.init(
    # set the wandb project where this run will be logged
    project="qwen-coder-dataset_py_c_test20_float32",

)


@dataclass
class CustomDataCollatorWithWeights(DataCollatorForLanguageModeling):
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    mlm: bool = False
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]], tokenizer: Optional[PreTrainedTokenizerBase] = None) -> Dict[str, torch.Tensor]:
        # Use the provided tokenizer or the one set during initialization
        if tokenizer is not None:
            self.tokenizer = tokenizer
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided either during initialization or when calling the collator.")

        # Extract weights and texts
        weights = [feature["weights"] for feature in features]
        texts = [feature["text"] for feature in features]
        
        # Tokenize the texts
        batch = self.tokenizer(
            texts,
            padding=self.padding,
            # max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch["labels"] = batch["input_ids"].clone()

        batch["labels"][batch["attention_mask"] == 0] = -100

        # shift the labels to the right
        # batch["labels"] = torch.roll(batch["labels"], 1, dims=1)

        # Convert weights to tensor and add to batch
        # batch["weights"] = torch.tensor(weights, dtype=torch.float)

        max_length = batch["input_ids"].shape[1]
        padded_weights = [w + [0] * (max_length - len(w)) for w in weights]
        batch["weights"] = torch.tensor(padded_weights, dtype=torch.float)
        #shift the weights to the right
        # batch["weights"] = torch.roll(batch["weights"], 1, dims=1)

        return batch
def tokenize_max(tokenizer, dataset):
  # Tokenize the dataset
  tokenized_dataset = tokenizer(dataset["text"])
  # Get the max length of the tokenized dataset
  max_length = max(map(len, tokenized_dataset["input_ids"]))
  print(f"Max length: {max_length}")

def get_weights(tokenizer, code, changes):
  be = tokenizer.encode_plus(code)
  tokens = be.data['input_ids']
  min_changed_tokens = 1
  if changes is None:
    weights = [1] * len(tokens)
  else:
    weights = [0] * len(tokens)
    for change in changes:
      char_start = change['char_start']
      char_start_idx = be.char_to_token(char_start)
      char_end = change['char_end']
      char_end_idx = be.char_to_token(char_end-1)
      for char_idx in range(char_start_idx, char_end_idx+1):
        weights[char_idx] = 1
      # if sum(weights) < min_changed_tokens: return None
      if len(tokens) - sum(weights) < min_changed_tokens: print("None")

  return tokens, weights

def token_weighted_loss(output, targets, weights):
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(output.view(-1, output.size(-1)), targets.view(-1))
    # flatten the weights
    weights = weights.view(-1)

    # print where wheights are not zero
    
    # print(len(loss), len(weights), len(weights[weights != 0]))
    loss = loss[weights != 0]
    # loss = weights * loss
    return loss.mean()

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        weights = inputs.pop("weights")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()
        loss = token_weighted_loss(shift_logits, shift_labels, shift_weights)

        return (loss, outputs) if return_outputs else loss
def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

if __name__ == "__main__":
  dataset = load_from_disk("datasets/") 
  model_dir = "../models/"

  model_name = "Qwen2.5-Coder-1.5B"#"../models_downloaded/codegen-2B-multi/"#"Salesforce/codegen-350m-multi"
  model_path = os.path.join(model_dir, model_name) # "facebook/incoder-6B"
  device_map = {"": 0}
  finetunes_model_name = "Qwen2.5-Coder-dataset_py_c_test20_float32"
  device_index = Accelerator().process_index
  device_map = {"": device_index}
  

  model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device_map)
  tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
  # tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  weights= []
  # dataset=dataset.add_column('weights',weights)
  print(model)
  for i in range(len(dataset)):
    _, weight = get_weights(tokenizer, dataset[i]["text"], dataset[i]["changes"]) #dataset[i]["changes"]
    # Add the weights to the dataset
    # print(weight)
    weights.append(weight)
  
  dataset=dataset.add_column('weights',weights)
  # dataset = dataset.add_column('weights', weights)
  dataset = dataset.remove_columns(["changes"])    
  # Shuffle the dataset["text"]
  dataset = dataset.shuffle(seed=42)
  # Devide the dataset into training and validation sets 
  dataset = dataset.train_test_split(test_size=0.2, seed=seed)
  tokenize_max(tokenizer, dataset["train"])
  tokenize_max(tokenizer, dataset["test"])
  print(len(dataset["train"]))
  print(len(dataset["test"]))
  
  data_collator = CustomDataCollatorWithWeights(tokenizer=tokenizer)
  peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16, # 64
    bias="none",
    task_type="CAUSAL_LM",
    target_modules = ["k_proj", "v_proj", "q_proj"]
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
    remove_unused_columns=False,
    seed=seed
    )

  trainer = CustomSFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length=512,
    tokenizer=tokenizer,
    dataset_text_field="text",
    # neftune_noise_alpha=5,
    args=trainingArgs,
    data_collator=data_collator,
    # accelerator_config={"dataloader_config": dataloader_config}
    )
  count_trainable_parameters(model)
  trainer.train() 
  wandb.finish()

