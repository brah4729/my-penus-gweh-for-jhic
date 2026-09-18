from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-0.8B",
    max_seq_length=4096,
    load_in_4bit=False,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none",
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-25")
dataset = load_dataset("json", data_files="data/tutoring_sft.jsonl", split="train")
def formatting_func(examples):
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in examples["messages"]
    ]
    return {"text": texts}
dataset = dataset.map(formatting_func, batched=True)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="out",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        dataset_text_field="text",
        max_seq_length=4096,
        logging_steps=10,
    ),
    train_dataset=dataset,
)
trainer.train()
model.save_pretrained_merged("school-assistant-merged", tokenizer)
model.save_pretrained_gguf("school-assistant-gguf", tokenizer, quantization_method="q4_k_m")