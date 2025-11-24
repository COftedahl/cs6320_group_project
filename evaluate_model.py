# evaluate_model.py
# Main evaluation script for sentiment analysis project
# Purpose: trains each model and tests them on test set

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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# How many samples to use - set to None to use all data
# We used smaller numbers while debugging
SAMPLE_SIZE = 5000  # Try 1000 first to make sure it works

TEST_DATA_PATH = FINAL_DATA_PATH + "test.csv"
VAL_DATA_PATH = FINAL_DATA_PATH + "val.csv"
RESULTS_DIR = "./results/"

# Make results folder if it doesn't already exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def load_data(sample_size=None):
    """loads train, val, and test data from our csv files"""
    print("Loading data...")
    parser = FileParser()
    
    # constantAdded=1 because labels are -1,0,1 and we need 0,1,2 for the model
    train_data = parser.parseFile(TRAINING_DATA_PATH, constantAdded=1)
    test_data = parser.parseFile(TEST_DATA_PATH, constantAdded=1)
    val_data = parser.parseFile(VAL_DATA_PATH, constantAdded=1)
    
    # If we're testing with smaller sample
    if sample_size:
        train_data = {
            TEXT_COLUMN_NAME: train_data[TEXT_COLUMN_NAME][:sample_size],
            VALUE_COLUMN_NAME: train_data[VALUE_COLUMN_NAME][:sample_size]
        }
        test_data = {
            TEXT_COLUMN_NAME: test_data[TEXT_COLUMN_NAME][:sample_size//8],
            VALUE_COLUMN_NAME: test_data[VALUE_COLUMN_NAME][:sample_size//8]
        }
        val_data = {
            TEXT_COLUMN_NAME: val_data[TEXT_COLUMN_NAME][:sample_size//8],
            VALUE_COLUMN_NAME: val_data[VALUE_COLUMN_NAME][:sample_size//8]
        }
    
    # Convert to huggingface dataset format
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)
    val_dataset = Dataset.from_dict(val_data)
    
    print(f"Loaded {len(train_dataset)} train, {len(test_dataset)} test, {len(val_dataset)} val samples")
    
    return train_dataset, test_dataset, val_dataset


def tokenize_data(train_ds, test_ds, val_ds, tokenizer):
    """Tokenize all the datasets with the given tokenizer"""
    
    def tokenize_fn(examples):
        # max_length=128 to keep it reasonable; can increase if needed
        return tokenizer(
            examples[TEXT_COLUMN_NAME], 
            padding="max_length", 
            truncation=True,
            max_length=128
        )
    
    train_tok = train_ds.map(tokenize_fn, batched=True)
    test_tok = test_ds.map(tokenize_fn, batched=True)
    val_tok = val_ds.map(tokenize_fn, batched=True)
    
    return train_tok, test_tok, val_tok


def get_metrics(eval_pred):
    """Compute accuracy for the trainer"""
    acc = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return acc.compute(predictions=preds, references=labels)


def train_model(model_name, model_path, train_ds, test_ds, val_ds, epochs=3):
    """
    Trains one model and evaluates it
    Returns dict with all the results
    """
    print("\n" + "="*50)
    print(f"Training: {model_name}")
    print(f"Model path: {model_path}")
    print("="*50)
    
    # Load pretrained model - num_labels=3 for neg/neu/pos
    # Ignore_mismatched_sizes because pretrained models might have different output size
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Tokenize everything
    train_tok, test_tok, val_tok = tokenize_data(train_ds, test_ds, val_ds, tokenizer)
    
    # Training config
    # Looked at huggingface docs for these settings
    train_args = TrainingArguments(
        output_dir=RESULTS_DIR + model_name + "_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        logging_steps=100,
        report_to="none",  # dont need wandb
    )
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,  # Validate on val set during training
        compute_metrics=get_metrics,
        data_collator=collator,
    )
    
    # Train it
    print("Starting training...")
    train_result = trainer.train()
    
    # Now evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_tok)
    
    # Obtain predictions so we can make confusion matrix
    print("Getting predictions...")
    pred_output = trainer.predict(test_tok)
    preds = np.argmax(pred_output.predictions, axis=-1)
    true_labels = pred_output.label_ids
    
    # sklearn classification report - gives us precision recall f1
    labels = ['Negative', 'Neutral', 'Positive']
    report = classification_report(true_labels, preds, target_names=labels, output_dict=True)
    report_text = classification_report(true_labels, preds, target_names=labels)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, preds)
    
    # Print results
    print(f"\nResults for {model_name}:")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test Loss: {test_results['eval_loss']:.4f}")
    print("\nClassification Report:")
    print(report_text)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save everything
    results = {
        "model_name": model_name,
        "model_source": model_path,
        "num_epochs": epochs,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "val_samples": len(val_ds),
        "training_loss": train_result.training_loss,
        "test_accuracy": test_results["eval_accuracy"],
        "test_loss": test_results["eval_loss"],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "timestamp": str(datetime.now())
    }
    
    # Save json
    with open(RESULTS_DIR + model_name + "_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + model_name + "_confusion_matrix.png")
    plt.close()
    
    print(f"Saved results to {RESULTS_DIR}")
    
    return results


def make_comparison_plot(all_results):
    """Makes bar chart comparing model accuracies"""
    names = [r["model_name"] for r in all_results]
    accs = [r["test_accuracy"] for r in all_results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accs)
    plt.ylabel('Test Accuracy')
    plt.title('Model Comparison')
    plt.ylim(0, 1)
    
    # Add labels on bars
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + "model_comparison.png")
    plt.close()
    print("Saved comparison plot")


def main():
    print("CS6320 Sentiment Analysis Evaluation")
    print("="*50)
    
    # Load data
    train_ds, test_ds, val_ds = load_data(sample_size=SAMPLE_SIZE)
    
    # All of the models we want to test
    models = [
        ("BERT_Multilingual", MODEL_OPTIONS.BERT_MULTILINGUAL.value),
        ("Sentiment_BERT", MODEL_OPTIONS.SENTIMENT_ANALYSIS_BERT.value),
        ("Twitter_RoBERTa", MODEL_OPTIONS.TWITTER_SENTIMENT_ANALYSIS_BERT.value),
    ]
    
    all_results = []
    
    for name, path in models:
        try:
            result = train_model(name, path, train_ds, test_ds, val_ds, epochs=3)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR with {name}: {e}")
            # Keep going with other models
            continue
    
    # Make comparison if we obtained multiple results
    if len(all_results) > 1:
        make_comparison_plot(all_results)
        
        # Save summary
        summary = {"models": []}
        for r in all_results:
            summary["models"].append({
                "name": r["model_name"],
                "accuracy": r["test_accuracy"],
                "f1": r["classification_report"]["macro avg"]["f1-score"]
            })
        with open(RESULTS_DIR + "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    print("\n" + "="*50)
    print("Done! Check ./results/ for output files")
    print("="*50)


if __name__ == "__main__":
    main()
