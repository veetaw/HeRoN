import torch
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, Trainer, T5Tokenizer, T5ForConditionalGeneration

df = pd.read_csv(
    'C:\\Users\\daisl\\PycharmProjects\\HeRoN\\dataset Reviewer\\game_scenarios_dataset_4.csv')

df['input'] = df['prompt'] + " " + df['response']

dataset = Dataset.from_pandas(df[['input', 'instructions']])

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

model.gradient_checkpointing_enable()
model.config.use_cache = False


def processes_function(examples):
    inputs = examples['input']
    targets = examples['instructions']
    model_inputs = tokenizer(inputs, max_length=512,
                             truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128,
                       truncation=True, padding="max_length")
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


tokenizer_dataset = dataset.map(
    processes_function,
    batched=True,
    remove_columns=dataset.column_names
)

split = tokenizer_dataset.train_test_split(test_size=0.2)
train_dataset = split['train']
eval_dataset = split['test']

training_args = TrainingArguments(
    output_dir="C:\\Users\\daisl\\PycharmProjects\\HeRoN\\args",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=3e-5,
    weight_decay=0.01,
    save_total_limit=3,
    logging_steps=10,
    fp16=True,
    dataloader_pin_memory=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained(
    "C:\\Users\\daisl\\PycharmProjects\\HeRoN\\flan-t5-ppo-pt")
tokenizer.save_pretrained(
    "C:\\Users\\daisl\\PycharmProjects\\HeRoN\\flan-t5-ppo-tok-pt")
