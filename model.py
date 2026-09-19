from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch
from pathlib import Path

model_id = "Qwen/Qwen3.5-0.8B"  # plain HF repo (CPU path, no Unsloth)
DATA_PATH = Path("data/datasets.jsonl")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run `uv run python build_dataset.py` first "
        "to generate the factual-QA portion of the dataset. Note that still "
        "only covers school facts -- Socratic tutoring dialogues need to be "
        "added separately before this dataset teaches the intended tutoring "
        "behavior."
    )

# 1. Load model on CPU
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cpu",
)

# 2. Apply LoRA
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. Load and format dataset (expects {"messages": [...]} per line, see build_dataset.py)
dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")


def formatting_func(examples):
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in examples["messages"]
    ]
    return {"text": texts}


dataset = dataset.map(formatting_func, batched=True)
print(f"Loaded {len(dataset)} training examples from {DATA_PATH}")

# 4. Train
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=SFTConfig(
        output_dir="out",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        dataset_text_field="text",
        max_length=2048,
        logging_steps=10,
        bf16=True,
        use_cpu=True,
        report_to="none",
    ),
    train_dataset=dataset,
)
trainer.train()

# 5. Save merged model
merged = model.merge_and_unload()
merged.save_pretrained("school-assistant-merged")
tokenizer.save_pretrained("school-assistant-merged")
