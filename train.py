import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Check if a compatible GPU is available and set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load your dataset
df = pd.read_csv('bearish_bullish_neutral_cleaned_5_0.1.csv')  # Replace with your dataset path

# Check and clean the label column
df = df.dropna(subset=['text', 'label'])  # Drop rows with missing texts or labels
df['label'] = df['label'].astype(str)     # Ensure labels are strings

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

# Reset indices to avoid KeyError during indexing in Dataset class
train_texts = train_texts.reset_index(drop=True)
val_texts = val_texts.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

# Ensure that all texts are strings
train_texts = train_texts.dropna().astype(str).tolist()
val_texts = val_texts.dropna().astype(str).tolist()

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Define a PyTorch Dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the datasets
train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

# Define the compute_metrics function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Best hyperparameters found by Optuna
best_learning_rate = 1.9689229956268363e-05
best_batch_size = 8
best_weight_decay = 0.00020105322157003673

# Set a lower learning rate
low_learning_rate = best_learning_rate * 0.1  # Adjust this factor as needed (e.g., 0.01)

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)

# Set training arguments with the lower learning rate
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=15,              # Number of training epochs
    per_device_train_batch_size=best_batch_size,   # Best batch size for training
    per_device_eval_batch_size=best_batch_size,    # Best batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=best_weight_decay,  # Best weight decay rate
    logging_dir='./logs',            # Directory to store logs during training
    logging_steps=10,                # How often to log training progress (every 10 steps)
    eval_strategy="epoch",           # Evaluation strategy, set to evaluate at the end of each epoch
    fp16=True,                       # Enable FP16 mixed precision for faster training on compatible hardware
    learning_rate=low_learning_rate, # Lower learning rate
    max_grad_norm=0.5,               # Apply gradient clipping to stabilize training
    lr_scheduler_type="cosine_with_restarts" # Use a learning rate scheduler to dynamically adjust the learning rate
)

# Initialize the Trainer with adjusted training arguments
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model using the lower learning rate
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
