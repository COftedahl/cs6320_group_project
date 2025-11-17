import torch.nn as nn
import torch.nn.functional as F
from constants import TEXT_COLUMN_NAME, VALUE_COLUMN_NAME
from transformers import pipeline, infer_device, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
import numpy as np

# Reference: https://huggingface.co/docs/transformers/quicktour

class LanguageModel(): 

  def __init__(self, modelSource): 
    self.modelSource = modelSource
    self.device = infer_device()
    self.pipe = pipeline("text-classification", model=modelSource, device=self.device)
    self.model = AutoModelForSequenceClassification.from_pretrained(modelSource, num_labels=2)
    self.tokenizer = AutoTokenizer.from_pretrained(modelSource)
    self.collocator = DataCollatorWithPadding(tokenizer=self.tokenizer)

  def train(self, trainData, testData, trainingArgs = TrainingArguments(
      output_dir="Data",
      learning_rate=2e-5,
      per_device_train_batch_size=8,
      per_device_eval_batch_size=8,
      num_train_epochs=2,
      push_to_hub=False,
      label_names=[VALUE_COLUMN_NAME]
  )): 
    
    def tokenizeTrainDataset(dataset):
      return self.tokenizer(dataset[TEXT_COLUMN_NAME])
    def tokenizeTestDataset(dataset):
      return self.tokenizer(dataset[VALUE_COLUMN_NAME])
    tokenizedTrainData = trainData.map(tokenizeTrainDataset, batched=True)
    tokenizedTestData = testData.map(tokenizeTestDataset, batched=True)
    print("Train:", tokenizedTrainData)
    print("Test: ", tokenizedTestData)

    def compute_loss_func(outputs, labels, num_items_in_batch=None):
      """
      Compute cross-entropy loss for transformer outputs.

      :param outputs: Model output from Hugging Face transformers (logits or dict with 'logits')
      :param labels: Tensor of shape (batch_size, seq_len) with target token IDs
      :param num_items_in_batch: Optional, number of valid items for averaging
      :return: Scalar loss tensor
      """
      print("Outputs:", outputs, "Labels: ", labels, "Num items in batch: ", num_items_in_batch)
      try:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        # Extract logits if outputs is a dict or ModelOutput
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        if logits.ndim != 2:
          raise ValueError(f"Expected logits of shape (batch, seq_len, vocab), got {logits.shape}")

        # Align shapes for CrossEntropyLoss: (batch*seq_len, vocab_size) vs (batch*seq_len)
        vocab_size = logits.size(-1)
        loss = loss_fn(
          logits.view(-1, vocab_size),
          labels.view(-1)
        )

        # Normalize if requested
        if self.reduction == "mean":
          if num_items_in_batch is None:
            # Default: average over non-ignored tokens
            valid_tokens = (labels != loss_fn.ignore_index).sum().item()
            num_items_in_batch = max(valid_tokens, 1)
          loss = loss / num_items_in_batch

        return loss

      except Exception as e:
        raise RuntimeError(f"Error computing loss: {e}")

    def cross_entropy_loss(predictions, targets, padding_mask):
      # Compute cross-entropy loss
      loss = F.cross_entropy(predictions, targets, reduction='none')
      # Apply padding mask to ignore padded tokens
      loss = loss * padding_mask
      # Compute mean loss over valid tokens
      return loss.sum() / padding_mask.sum()
    
    metric = evaluate.load("confusion_matrix")

    def compute_metrics(eval_preds):
      logits, labels = eval_preds
      predictions = np.argmax(logits, axis=-1)
      return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
      model=self.model,
      args=trainingArgs,
      train_dataset=tokenizedTrainData,
      eval_dataset=tokenizedTestData,
      tokenizer=self.tokenizer,
      data_collator=self.collocator,
      compute_loss_func=compute_loss_func,
      compute_metrics=compute_metrics, 
    )
    trainer.train()

  def test(self, data): 
    return self.pipe(data)