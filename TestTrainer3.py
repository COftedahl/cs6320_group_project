import torch.nn as nn
import torch.nn.functional as F
from FileParser import FileParser
from constants import MODEL_OPTIONS, TEXT_COLUMN_NAME, TRAINING_DATA_PATH, VALUE_COLUMN_NAME
from transformers import pipeline, infer_device, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
import numpy as np
from datasets import Dataset

# Reference: https://huggingface.co/docs/transformers/quicktour

class LanguageModel(): 

  def __init__(self, modelSource): 
    self.modelSource = modelSource
    self.device = infer_device()
    self.pipe = pipeline("text-classification", model=modelSource, device=self.device)
    self.model = AutoModelForSequenceClassification.from_pretrained(modelSource)
    self.tokenizer = AutoTokenizer.from_pretrained(modelSource)
    self.collocator = DataCollatorWithPadding(tokenizer=self.tokenizer)

  def train(self, trainData, testData, trainingArgs = TrainingArguments(
    output_dir="Data",
    eval_strategy="epoch",
    push_to_hub=False,
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

if __name__ == '__main__':
  fileParser = FileParser()
  trainingData = fileParser.parseFile(TRAINING_DATA_PATH, constantAdded=1)
  reducedTrainingData = {TEXT_COLUMN_NAME: trainingData[TEXT_COLUMN_NAME][0:100], VALUE_COLUMN_NAME: trainingData[VALUE_COLUMN_NAME][0:100]}
  # print(reducedTrainingData)
  # trainingDataSet = Dataset.from_dict(trainingData)
  trainingDataSet = Dataset.from_dict(reducedTrainingData)
  # print(trainingDataSet)
  languageModel = LanguageModel(MODEL_OPTIONS.SENTIMENT_ANALYSIS_BERT.value)
  languageModel.train(trainingDataSet, trainingDataSet)