from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate

# Load dataset and tokenizer
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "MarieAngeA13/Sentiment-Analysis-BERT"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenize the dataset
def tokenize_function(example):
  return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define the model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# Define training arguments
training_args = TrainingArguments(
output_dir="test-trainer",
eval_strategy="epoch",
learning_rate=2e-5,
per_device_train_batch_size=16,
per_device_eval_batch_size=16,
num_train_epochs=3,
weight_decay=0.01,
logging_dir="./logs",
logging_steps=10,
save_strategy="epoch"
)

# Define metrics for evaluation
metric = evaluate.load("glue", "mrpc")

def compute_metrics(eval_preds):
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)

# Initialize the Trainer
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_datasets["train"],
  eval_dataset=tokenized_datasets["validation"],
  tokenizer=tokenizer,
  data_collator=data_collator,
  compute_metrics=compute_metrics
)

# Train the model
trainer.train()