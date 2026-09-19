from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch
model_id = "Qwen/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
dataset = load_dataset("json", data_files="data/datasets.jsonl", split="train")
def formatting_func(examples):
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in examples["messages"]
    ]
    return {"text": texts}
dataset = dataset.map(formatting_func, batched=True)
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="out",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        dataset_text_field="text",
        max_seq_length=2048,
        logging_steps=10,
        bf16=True,
        use_cpu=True,
        report_to="none",
    ),
    train_dataset=dataset,
)
trainer.train()
merged = model.merge_and_unload()
merged.save_pretrained("school-assistant-merged")
tokenizer.save_pretrained("school-assistant-merged")