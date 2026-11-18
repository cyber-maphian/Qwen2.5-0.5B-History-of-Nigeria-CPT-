---
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen2.5-0.5B
pipeline_tag: text-generation
language:
- en
datasets:
- Cyber-Maphian/History_of_Nigeria
tags:
- lora
- qwen2.5
- continued-pretraining
- domain-adaptation
- nigeria
- history
---

# Qwen2.5-0.5B-Nigeria-History-Base-LoRA

This repository contains a LoRA adapter for [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B), domain-adapted via continued causal language modeling on Nigerian historical texts ([`Cyber-Maphian/History_of_Nigeria`](https://huggingface.co/datasets/Cyber-Maphian/History_of_Nigeria)).

> **Note:** This is a **domain-adapted base model** (next-token predictor), not an instruction-tuned chat assistant. It is designed to serve as an intermediate foundation checkpoint for downstream Supervised Fine-Tuning (SFT) on Nigerian history Q&A and instruction datasets.

---

## Model Details

- **Developer:** [Cyber-Maphian](https://huggingface.co/Cyber-Maphian)
- **Model Name:** `Cyber-Maphian/Qwen2.5-0.5B-History-of-Nigeria-CPT`
- **Base Model:** [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) (0.5B parameters)
- **Fine-Tuning Method:** Parameter-Efficient Fine-Tuning (LoRA via `peft`)
- **Task:** Causal Language Modeling / Continued Pre-Training (`CAUSAL_LM`)
- **Language:** English
- **License:** Apache 2.0 (matching the base Qwen2.5-0.5B license)

---

## Intended Uses & Limitations

### Intended Use
- **Intermediate Base Checkpoint:** Serving as the domain-enriched starting point for subsequent instruction fine-tuning (SFT) or preference alignment (DPO/ORPO) on Nigerian history tasks.
- **Historical Text Continuation:** Completing historical passages, timelines, and narrative text related to Nigeria.

### Out-of-Scope Use & Limitations
- **Not an Instruction-Following Assistant:** Because this adapter was trained on raw causal text completion without a chat template, prompting it with direct questions triggers document-style continuations rather than concise conversational answers.
- **Corpus Artifacts & Hallucinations:** As a 0.5B parameter model trained on historical corpus chunks, it can repeat chapter headers/page numbers from source documents, generate run-on continuations, or hallucinate dates and political figures. Do not rely on it as an authoritative historical reference without verification.

---

## Training Data

The adapter was trained on the [`Cyber-Maphian/History_of_Nigeria`](https://huggingface.co/datasets/Cyber-Maphian/History_of_Nigeria) dataset:
- **Train Split:** ~1.5k examples
- **Validation Split:** 196 examples
- **Test Split:** 197 examples

---

## Training Procedure

### LoRA Configuration
The adapter targets both attention and MLP projection layers across the Qwen2.5-0.5B architecture:

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none"
)
```
### Training Logs & Checkpoint Selection

Training was evaluated every 50 steps using validation cross-entropy loss (`eval_loss`), with `load_best_model_at_end=True` selecting the checkpoint at **Step 350** prior to validation loss divergence:

| Step | Training Loss | Validation Loss |
| :--- | :---: | :---: |
| 50 | 2.3644 | 2.0954 |
| 100 | 2.4235 | 2.0565 |
| 150 | 2.2683 | 2.0307 |
| 200 | 2.3226 | 2.0173 |
| 250 | 2.2013 | 2.0122 |
| 300 | 2.2392 | 2.0087 |
| **350 (Best)** | **2.1246** | **2.0059** |
| 400 | 2.1892 | 2.0082 |
| 450 | 2.1885 | 2.0097 |

---

### Qualitative Comparison (Base vs. Fine-Tuned)

Because both models operate as raw causal next-token predictors without a chat template, outputs reflect document continuation rather than direct Q&A:

| Prompt | Original `Qwen/Qwen2.5-0.5B` | Fine-Tuned Adapter (This Model) |
| :--- | :--- | :--- |
| `when did nigeria gain independence` | *"we start writing the constitution?? 1960"* | *"the nigeria independence? 1960 – 1964 337 Nigeria gained independence on 1 October 1960. 1960 marked the beginning of a 10-year period of political and economic transition, as Nigeria became increasingly divided along ethnic and regional lines..."* |
| `who is bola ahmed tinubu` | *"HumanA person who is a famous footballer of which country Answer: South Africa"* | *"te? who is the prime minister of nigeria? who is the president of nigeria? ... CHAPTER 4 The state of Nigeria: a short history 350 Nigeria, whose population is less than 100 million, was founded in 1914 as the Colony of Nigeria..."* |


### How To Use Model
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-0.5B"
adapter_id = "Cyber-Maphian/Qwen2.5-0.5B-History-of-Nigeria-CPT" 

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16
).to("cuda" if torch.cuda.is_available() else "cpu")

# 2. Wrap with PEFT adapter safely
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

# 3. Test inference
prompt = "Nigeria gained independence on"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
