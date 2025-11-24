# quick_test.py
# Run this first to make sure everything works before doing full training
# (Uses small sample so it finishes quickly)

from FileParser import FileParser
from constants import (
    TEXT_COLUMN_NAME, TRAINING_DATA_PATH, MODEL_OPTIONS, 
    VALUE_COLUMN_NAME, FINAL_DATA_PATH
)
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, 
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset
import evaluate
import numpy as np
from sklearn.metrics import classification_report
import os
import shutil

# Small samples for quick test
TRAIN_SAMPLES = 500
TEST_SAMPLES = 100

TEST_DATA_PATH = FINAL_DATA_PATH + "test.csv"


def main():
    print("="*50)
    print("Quick Test - checking if pipeline works")
    print("="*50)
    
    # Load small amount of data
    print("\n1. Loading data...")
    parser = FileParser()
    train_data = parser.parseFile(TRAINING_DATA_PATH, constantAdded=1)
    test_data = parser.parseFile(TEST_DATA_PATH, constantAdded=1)
    
    # Take only first few samples
    train_data = {
        TEXT_COLUMN_NAME: train_data[TEXT_COLUMN_NAME][:TRAIN_SAMPLES],
        VALUE_COLUMN_NAME: train_data[VALUE_COLUMN_NAME][:TRAIN_SAMPLES]
    }
    test_data = {
        TEXT_COLUMN_NAME: test_data[TEXT_COLUMN_NAME][:TEST_SAMPLES],
        VALUE_COLUMN_NAME: test_data[VALUE_COLUMN_NAME][:TEST_SAMPLES]
    }
    
    print(f"   Got {len(train_data[TEXT_COLUMN_NAME])} train samples")
    print(f"   Got {len(test_data[TEXT_COLUMN_NAME])} test samples")
    
    # Make sure labels look right
    print("\n2. Label distribution in train set:")
    labels = train_data[VALUE_COLUMN_NAME]
    print(f"   0 (negative): {labels.count(0)}")
    print(f"   1 (neutral): {labels.count(1)}")
    print(f"   2 (positive): {labels.count(2)}")
    
    # Show example
    print("\n3. Example data point:")
    print(f"   Text: {train_data[TEXT_COLUMN_NAME][0][:80]}...")
    print(f"   Label: {train_data[VALUE_COLUMN_NAME][0]}")
    
    # Create datasets
    train_ds = Dataset.from_dict(train_data)
    test_ds = Dataset.from_dict(test_data)
    
    # Attempt to load model
    print("\n4. Loading model and tokenizer...")
    model_path = MODEL_OPTIONS.SENTIMENT_ANALYSIS_BERT.value
    print(f"   Using: {model_path}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Tokenize
    def tok_fn(examples):
        return tokenizer(examples[TEXT_COLUMN_NAME], padding="max_length", 
                        truncation=True, max_length=128)
    
    train_tok = train_ds.map(tok_fn, batched=True)
    test_tok = test_ds.map(tok_fn, batched=True)
    print("   Tokenization works!")
    
    # Do quick training
    print("\n5. Training for 1 epoch (this might take a minute)...")
    
    acc_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return acc_metric.compute(predictions=preds, references=labels)
    
    args = TrainingArguments(
        output_dir="./test_output_temp",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    trainer.train()
    
    # Perform testing
    print("\n6. Testing...")
    results = trainer.evaluate(test_tok)
    print(f"   Accuracy: {results['eval_accuracy']:.4f}")
    print(f"   Loss: {results['eval_loss']:.4f}")
    
    # Obtain classification report
    pred_out = trainer.predict(test_tok)
    preds = np.argmax(pred_out.predictions, axis=-1)
    
    print("\n7. Classification Report:")
    print(classification_report(pred_out.label_ids, preds, 
          target_names=['Negative', 'Neutral', 'Positive']))
    
    # Clean up temp folder
    if os.path.exists("./test_output_temp"):
        shutil.rmtree("./test_output_temp")
    
    print("\n" + "="*50)
    print("TEST PASSED - everything works!")
    print("Now you can run evaluate_model.py for full training")
    print("="*50)


if __name__ == "__main__":
    main()
