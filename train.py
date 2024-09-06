import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import optuna

# Check if a compatible GPU is available and set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load your dataset
df = pd.read_csv('path_to_your_dataset.csv')  # Replace with your dataset path

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

# Hyperparameter optimization function using Optuna
def objective(trial):
    # Suggest hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)

    # Load the BERT model for sequence classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # number of training epochs
        per_device_train_batch_size=batch_size,   # batch size for training
        per_device_eval_batch_size=batch_size,    # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,       # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        eval_strategy="epoch",           # use eval_strategy instead of evaluation_strategy
        fp16=True,                       # Enable mixed precision
        learning_rate=learning_rate      # Set the learning rate
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()
    return eval_result['eval_accuracy']

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Print best hyperparameters
print("Best hyperparameters: ", study.best_params)
print("Best accuracy: ", study.best_value)



"""
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)
"""
