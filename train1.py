import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaModel

# Check if a compatible GPU is available and set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load your dataset
df = pd.read_csv('proposal_df_summ.csv')  # Replace with your dataset path

# Check and clean the 'proposal_type' and 'category' columns
df['proposal_type'] = df['proposal_type'].astype(str)
df['category'] = df['category'].astype(str)

# Encode both 'proposal_type' and 'category' labels
proposal_type_encoder = LabelEncoder()
category_encoder = LabelEncoder()

df['proposal_type_label'] = proposal_type_encoder.fit_transform(df['proposal_type'])
df['category_label'] = category_encoder.fit_transform(df['category'])

# Print label mappings for both
proposal_type_mapping = {index: label for index, label in enumerate(proposal_type_encoder.classes_)}
category_mapping = {index: label for index, label in enumerate(category_encoder.classes_)}
print("Proposal Type Mapping:", proposal_type_mapping)
print("Category Mapping:", category_mapping)

# Split the dataset into training and validation sets
train_texts, val_texts, train_proposal_labels, val_proposal_labels, train_category_labels, val_category_labels = train_test_split(
    df['proposal'], df['proposal_type_label'], df['category_label'], test_size=0.2, random_state=42
)

# Reset indices to avoid KeyError during indexing in Dataset class
train_texts = train_texts.reset_index(drop=True)
val_texts = val_texts.reset_index(drop=True)
train_proposal_labels = train_proposal_labels.reset_index(drop=True)
val_proposal_labels = val_proposal_labels.reset_index(drop=True)
train_category_labels = train_category_labels.reset_index(drop=True)
val_category_labels = val_category_labels.reset_index(drop=True)

# Ensure all texts are strings
train_texts = train_texts.dropna().astype(str).tolist()
val_texts = val_texts.dropna().astype(str).tolist()

# Initialize the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenize the dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Define a PyTorch Dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, proposal_labels, category_labels):
        self.encodings = encodings
        self.proposal_labels = proposal_labels
        self.category_labels = category_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['proposal_labels'] = torch.tensor(self.proposal_labels[idx])
        item['category_labels'] = torch.tensor(self.category_labels[idx])
        return item

    def __len__(self):
        return len(self.proposal_labels)

# Create the datasets with two labels
train_dataset = CustomDataset(train_encodings, train_proposal_labels, train_category_labels)
val_dataset = CustomDataset(val_encodings, val_proposal_labels, val_category_labels)

# Define a multitask learning model using RoBERTa
class MultiTaskRobertaModel(torch.nn.Module):
    def __init__(self, model_name, num_proposal_labels, num_category_labels):
        super(MultiTaskRobertaModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.proposal_classifier = torch.nn.Linear(self.roberta.config.hidden_size, num_proposal_labels)
        self.category_classifier = torch.nn.Linear(self.roberta.config.hidden_size, num_category_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, proposal_labels=None, category_labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # get [CLS] token output
        
        # Proposal (main sentiment) classification head
        proposal_logits = self.proposal_classifier(pooled_output)
        
        # Category (secondary sentiment) classification head
        category_logits = self.category_classifier(pooled_output)
        
        loss = None
        if proposal_labels is not None and category_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            proposal_loss = loss_fct(proposal_logits, proposal_labels)
            category_loss = loss_fct(category_logits, category_labels)
            loss = proposal_loss + category_loss

        return {
            'loss': loss,
            'proposal_logits': proposal_logits,
            'category_logits': category_logits
        }

# Instantiate the multitask model
model = MultiTaskRobertaModel(
    model_name='roberta-base', 
    num_proposal_labels=len(proposal_type_encoder.classes_),
    num_category_labels=len(category_encoder.classes_)
).to(device)

# Define the compute_metrics function for multitask learning
def compute_metrics(pred):
    proposal_labels = pred.label_ids[0]
    category_labels = pred.label_ids[1]
    
    proposal_preds = pred.predictions[0].argmax(-1)
    category_preds = pred.predictions[1].argmax(-1)
    
    # Proposal (main sentiment) metrics
    proposal_precision, proposal_recall, proposal_f1, _ = precision_recall_fscore_support(proposal_labels, proposal_preds, average='weighted')
    proposal_acc = accuracy_score(proposal_labels, proposal_preds)
    
    # Category (secondary sentiment) metrics
    category_precision, category_recall, category_f1, _ = precision_recall_fscore_support(category_labels, category_preds, average='weighted')
    category_acc = accuracy_score(category_labels, category_preds)
    
    return {
        'proposal_accuracy': proposal_acc,
        'proposal_f1': proposal_f1,
        'proposal_precision': proposal_precision,
        'proposal_recall': proposal_recall,
        'category_accuracy': category_acc,
        'category_f1': category_f1,
        'category_precision': category_precision,
        'category_recall': category_recall
    }

# Define a data collator for multitask learning
def data_collator(features):
    input_ids = torch.stack([f['input_ids'] for f in features])
    attention_mask = torch.stack([f['attention_mask'] for f in features])
    proposal_labels = torch.stack([f['proposal_labels'] for f in features])
    category_labels = torch.stack([f['category_labels'] for f in features])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'proposal_labels': proposal_labels,
        'category_labels': category_labels
    }

# Best hyperparameters (from previous tuning)
best_learning_rate = 1.9689229956268363e-05
best_batch_size = 8
best_weight_decay = 0.00020105322157003673

# Set a lower learning rate for fine-tuning
low_learning_rate = best_learning_rate * 0.1

# Training arguments for multitask learning
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
    # Removed metric_for_best_model, as eval_loss is tracked by default
    greater_is_better=False  # We want lower eval_loss
)

# Initialize the Trainer with multitask support
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]  # Stop if no improvement in 10 evals
)

# Train the model
trainer.train()

# Save the final model and tokenizer
model.save_pretrained('./trained_multitask_model')
tokenizer.save_pretrained('./trained_multitask_model')

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
