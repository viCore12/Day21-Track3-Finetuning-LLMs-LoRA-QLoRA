# Lab 21 — External Links (Option B Submission)

**Học viên**: Lưu Lương Vi Nhân - 2A202600120
**Repo link**: https://github.com/viCore12/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA
---

## HuggingFace Hub

| Adapter | URL |
|---------|-----|
| r=16 baseline (best rank, publicly deployed) | https://huggingface.co/Nhanvi282/lab21-qwen25-3b-vi-r16 |

---

## Weights & Biases — Loss Curves

| Run | URL |
|-----|-----|
| r=16 training (loss curve tracked real-time) | https://wandb.ai/nhanvi212/lab21-lora-rank-experiment/runs/gegylngh |

---

## Usage Example

```python
from peft import PeftModel
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B-bnb-4bit",
    max_seq_length=1024,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "Nhanvi282/lab21-qwen25-3b-vi-r16")
FastLanguageModel.for_inference(model)

prompt = "### Instruction:\nGiải thích LoRA cho người mới bắt đầu.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
