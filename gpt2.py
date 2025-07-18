import os
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, pipeline

# 1. Check data file
data_path = "custom_data.txt"
if not os.path.exists(data_path):
    raise FileNotFoundError("custom_data.txt not found in the current directory.")

# 2. Load dataset
dataset = load_dataset("text", data_files={"train": data_path})

# 3. Load tokenizer and set pad token
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 4. Tokenize dataset (input_ids + labels)
def tokenize_function(examples):
    result = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. Load GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# 6. Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=100,
    save_total_limit=1,
    logging_steps=10,
    prediction_loss_only=True,
    report_to=[]  # Avoid wandb or other integrations
)

# 7. Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# 8. Train model
trainer.train()

# 9. Generate text
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "Once upon a time"
output = generator(prompt, max_length=100, num_return_sequences=1)
print("\nGenerated Text:\n", output[0]["generated_text"])
