# LoRA Hyperparameters

LoRA hyperparameters used to fine-tune the pre-trained CodeLMs.

| Models | Batch Size | #Epoch | Learning Rate | Rank | LoRA α |
|--------|------------|--------|---------------|------|--------|
| CodeGen-350M-multi | 16 | 10 | 5e-4 | 64 | 16 |
| CodeGen-2B-multi   | 16 | 10 | 5e-4 | 64 | 16 |
| Incoder-6B           | 16 | 10 | 5e-4 | 64 | 16 |
| Qwen2.5-Coder-1.5B | 16 | 10 | 5e-4 | 8  | 16 |
| DeepSeek-Coder-V2-16B | 16 | 10 | 5e-4 | 16 | 16 |
