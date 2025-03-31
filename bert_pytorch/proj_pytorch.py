import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Define global variables
maxlen = 512  # Maximum length for input sequences
num_classes = 16  # Adjust this based on the number of target classes
model_name = 'bert-large-uncased'  # Model name for the tokenizer and BERT layer

# Load and preprocess data
data = pd.read_csv('./kaggle.csv')
data['type_index'] = pd.Categorical(data['type']).codes

# Clean text
def clean_text(text):
    import html
    import re
    text = html.unescape(text)
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'#', '', text)  # Remove hashtags
    return text

data['cleaned_text'] = data['text'].apply(clean_text)

# Split data
train, test = train_test_split(data, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, maxlen):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, 
            return_tensors='pt', 
            max_length=self.maxlen, 
            padding='max_length', 
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create Datasets and DataLoaders
train_dataset = TextDataset(train['cleaned_text'].tolist(), train['type_index'].tolist(), tokenizer, maxlen)
val_dataset = TextDataset(val['cleaned_text'].tolist(), val['type_index'].tolist(), tokenizer, maxlen)
test_dataset = TextDataset(test['cleaned_text'].tolist(), test['type_index'].tolist(), tokenizer, maxlen)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# Model Definition
class MBTIClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(MBTIClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MBTIClassifier(model_name, num_classes).to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*4)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training Loop
def train_model(model, train_loader, val_loader, test_loader, optimizer, scheduler, device, epochs=4):
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            optimizer.zero_grad()
            
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            # Print batch progress
            if batch_count % 10 == 0:
                print(f"Batch {batch_count}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")

        # Save the model after each epoch
        model_path = f"mbti_classifier_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

        # Validation
        print("Starting validation...")
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Test
        print("Starting testing...")
        test_preds, test_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                test_preds.extend(preds.cpu().tolist())
                test_labels.extend(labels.cpu().tolist())
        test_acc = accuracy_score(test_labels, test_preds)
        print(f"Test Accuracy for epoch {epoch + 1}: {test_acc:.4f}")
        print("-" * 50)

# Call the training function
train_model(model, train_loader, val_loader, test_loader, optimizer, scheduler, device)
# Train the model
#train_model(model, train_loader, val_loader, optimizer, scheduler, device)

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,  # Ensure this argument is included
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=4  # You can adjust the number of epochs as needed
)

# Save the model
torch.save(model.state_dict(), "mbti_classifier.pth")

