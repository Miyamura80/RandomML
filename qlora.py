
from transformers import (TrainingArguments, Trainer)

from peft import (LoraConfig, get_peft_model
                  prepare_model_for_kbit_training)

# Prep QLoRA model
model = prepare_model_for_kbit_training(
    pretrained_model)

# Set LoRA config
lora_config = LoraConfig(r=32, lora_alpha=32,
    target_modules=['q_proj', 'k_proj',
        'v_proj', 'dense'])

# Get PEFT model
peft_model = get_peft_model(
    model, lora_config)

# Training args
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir='./logs')

# Init Trainer
trainer = Trainer(
    model=peft_model,
args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset)

# Fine-tune
trainer.train()
