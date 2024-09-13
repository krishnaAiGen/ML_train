#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 02:33:44 2024

@author: krishnayadav
"""

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


# Check if a compatible GPU is available and set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load your dataset
df = pd.read_csv('5_20_summarize_mae.csv')  # Replace with your dataset path

# Check and clean the label column
df = df.dropna(subset=['text', 'percent_price'])  # Drop rows with missing texts or prices
df['percent_price'] = df['percent_price'].astype(float)   # Ensure prices are floats

# Split the dataset into training and validation sets
train_texts, val_texts, train_prices, val_prices = train_test_split(
    df['text'], df['percent_price'], test_size=0.2, random_state=42)

# Reset indices to avoid KeyError during indexing in Dataset class
train_texts = train_texts.reset_index(drop=True)
val_texts = val_texts.reset_index(drop=True)
train_prices = train_prices.reset_index(drop=True)
val_prices = val_prices.reset_index(drop=True)

# Ensure that all texts are strings
train_texts = train_texts.dropna().astype(str).tolist()
val_texts = val_texts.dropna().astype(str).tolist()

# Initialize the RoBERTa tokenizer for large model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenize the dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Define a PyTorch Dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, prices):
        self.encodings = encodings
        self.prices = prices

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.prices[idx], dtype=torch.float)  # Continuous value for regression
        return item

    def __len__(self):
        return len(self.prices)

# Create the datasets
train_dataset = CustomDataset(train_encodings, train_prices)
val_dataset = CustomDataset(val_encodings, val_prices)

# Define the compute_metrics function for evaluation using MAE
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.flatten()
    mae = mean_absolute_error(labels, preds)
    return {
        'mae': mae
    }

# Custom model for regression using RoBERTa large
class RobertaForRegression(torch.nn.Module):
    def __init__(self, model_name='roberta-base'):
        super(RobertaForRegression, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.regressor = torch.nn.Linear(self.roberta.config.hidden_size, 1)  # Regression head

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        logits = self.regressor(outputs.pooler_output)  # Use the pooled output for regression
        loss = None
        if labels is not None:
            loss = torch.nn.functional.l1_loss(logits.squeeze(), labels)  # MAE (L1) loss for regression
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

# Initialize the model with roberta-large
model = RobertaForRegression().to(device)

# Best hyperparameters found by Optuna or predefined
best_learning_rate = 1.9689229956268363e-05
best_batch_size = 4  # Consider reducing if you face memory issues
best_weight_decay = 0.00020105322157003673

# Set a lower learning rate
low_learning_rate = best_learning_rate * 0.1  # Adjust this factor as needed (e.g., 0.01)

# Set training arguments to avoid saving intermediate checkpoints
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=200,             # Number of training epochs
    per_device_train_batch_size=best_batch_size,   # Best batch size for training, reduce if OOM
    per_device_eval_batch_size=best_batch_size,    # Best batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=best_weight_decay,  # Best weight decay rate
    logging_dir='./logs',            # Directory to store logs during training
    logging_steps=10,                # How often to log training progress (every 10 steps)
    eval_strategy="epoch",           # Evaluation strategy, set to evaluate at the end of each epoch
    fp16=True,                       # Enable FP16 mixed precision for faster training on compatible hardware
    learning_rate=low_learning_rate, # Lower learning rate
    max_grad_norm=0.5,               # Apply gradient clipping to stabilize training
    lr_scheduler_type="cosine_with_restarts",  # Use a learning rate scheduler to dynamically adjust the learning rate
    save_strategy="no",              # Disable checkpoint saving
)

# Initialize the Trainer with adjusted training arguments
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the final model at the end
trainer.save_model('./trained_model')

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)