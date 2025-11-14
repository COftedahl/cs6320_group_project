from constants import TEXT_COLUMN_NAME, VALUE_COLUMN_NAME
from transformers import pipeline, infer_device, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding

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
      learning_rate=2e-5,
      per_device_train_batch_size=8,
      per_device_eval_batch_size=8,
      num_train_epochs=2,
      push_to_hub=False,
  )): 
    
    def tokenizeTrainDataset(dataset):
      return self.tokenizer(dataset[TEXT_COLUMN_NAME])
    def tokenizeTestDataset(dataset):
      return self.tokenizer(dataset[VALUE_COLUMN_NAME])
    tokenizedTrainData = trainData.map(tokenizeTrainDataset, batched=True)
    tokenizedTestData = testData.map(tokenizeTestDataset, batched=True)

    trainer = Trainer(
      model=self.model,
      args=trainingArgs,
      train_dataset=tokenizedTrainData,
      eval_dataset=tokenizedTestData,
      tokenizer=self.tokenizer,
      data_collator=self.collocator,
    )
    trainer.train()

  def test(self, data): 
    return self.pipeline(data)