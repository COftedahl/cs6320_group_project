from constants import TEXT_COLUMN_NAME
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
import numpy as np
import torch
# Reference: https://huggingface.co/docs/transformers/quicktour

class LanguageModel(): 

  def __init__(self, modelSource): 
    self.modelSource = modelSource
    self.device = 0 if torch.cuda.is_available() else -1

    self.model = AutoModelForSequenceClassification.from_pretrained(modelSource)
    self.tokenizer = AutoTokenizer.from_pretrained(modelSource)

    if self.tokenizer.pad_token is None and hasattr(self.tokenizer, "eos_token"):
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.model.config.pad_token_id = self.model.config.eof_token_id

    self.pipe = pipeline("text-classification", model=modelSource, tokenizer = self.tokenizer, device=self.device)
    
    self.collocator = DataCollatorWithPadding(tokenizer=self.tokenizer)

  def train(self, trainData, testData, trainingArgs = TrainingArguments(
    output_dir="Data",
    eval_strategy="epoch",
    push_to_hub=False,
    save_strategy="no",
  )): 
    
    def tokenizeDataset(dataset):
      return self.tokenizer(dataset[TEXT_COLUMN_NAME], padding="max_length", truncation=True)
    tokenizedTrainData = trainData.map(tokenizeDataset, batched=True)
    tokenizedTestData = testData.map(tokenizeDataset, batched=True)
    print("Train:", tokenizedTrainData)
    print("Test: ", tokenizedTestData)
    
    # metric = evaluate.load("accuracy")
    metric = evaluate.load("confusion_matrix")

    def compute_metrics(eval_pred):
      logits, labels = eval_pred
      # convert the logits to their predicted class
      predictions = np.argmax(logits, axis=-1)
      return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
      model=self.model,
      args=trainingArgs,
      train_dataset=tokenizedTrainData,
      eval_dataset=tokenizedTrainData,
      compute_metrics=compute_metrics,
    )
    print("trainer train_dataset:")
    print(trainer.train_dataset)
    print(trainer.train_dataset[0])
    trainer.train()

  def test(self, data): 
    return self.pipe(data)