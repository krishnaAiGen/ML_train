import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Check if a compatible GPU is available and set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load your dataset
df = pd.read_csv('cleaned_bullish_proposal31oct.csv')  # Replace with your dataset path

# Check and clean the label column
# df = df.dropna(subset=['text', 'label'])  # Drop rows with missing texts or labels
# df['label'] = df['label'].astype(str)     # Ensure labels are strings

# Encode labels
label_encoder = LabelEncoder()
df['sentiment_new'] = label_encoder.fit_transform(df['sentiment_new'])
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
print("Label Mapping:", label_mapping)

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['summary'], df['sentiment_new'], test_size=0.2, random_state=42)

# Reset indices to avoid KeyError during indexing in Dataset class
train_texts = train_texts.reset_index(drop=True)
val_texts = val_texts.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

# Ensure that all texts are strings
train_texts = train_texts.dropna().astype(str).tolist()
val_texts = val_texts.dropna().astype(str).tolist()

# Initialize the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

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

# Define the compute_metrics function for evaluation, including confusion matrix
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)
    
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

# Load the RoBERTa model for sequence classification
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3).to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=50,
    per_device_train_batch_size=best_batch_size,
    per_device_eval_batch_size=best_batch_size,
    warmup_steps=500,
    weight_decay=best_weight_decay,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",  # Save at each epoch
    save_total_limit=1,  # Keep only the best checkpoint
    fp16=True,
    learning_rate=low_learning_rate,
    max_grad_norm=0.5,
    lr_scheduler_type="cosine_with_restarts",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# Initialize the Trainer with adjusted training arguments and early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if no improvement in 3 evals
)

# Train the model using the lower learning rate
trainer.train()

# Save the final model at the end
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
