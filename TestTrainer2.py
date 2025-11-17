from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from datasets import Dataset

from FileParser import FileParser
import constants

# Load dataset and tokenizer
raw_datasets = load_dataset("glue", "mrpc")
raw2_datasets = load_dataset("csv", constants.TRAINING_DATA_PATH)
fileParser = FileParser()
trainingData = fileParser.parseFile(constants.TRAINING_DATA_PATH)
trainingDataSet = Dataset.from_dict(trainingData)
checkpoint = "MarieAngeA13/Sentiment-Analysis-BERT"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

print(raw_datasets)
print("Raw: ", raw_datasets["train"]["sentence1"])
print(raw2_datasets)
print(trainingDataSet)
print("Mine: ", trainingDataSet[constants.TEXT_COLUMN_NAME])
print("Mine: ", trainingDataSet[constants.VALUE_COLUMN_NAME])

# Tokenize the dataset
def tokenize_function(example):
  return tokenizer(example[constants.TEXT_COLUMN_NAME], example[constants.VALUE_COLUMN_NAME], truncation=True, padding=True)

tokenized_datasets = trainingDataSet.map(tokenize_function, batched=True)
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
metric = evaluate.load("confusion_matrix")

def compute_metrics(eval_preds):
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)

# Initialize the Trainer
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_datasets,
  eval_dataset=tokenized_datasets,
  tokenizer=tokenizer,
  data_collator=data_collator,
  compute_metrics=compute_metrics
)

# Train the model
trainer.train()