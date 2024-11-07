import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

# Constants remain the same except for batch size adjustment
SEED = 42
MAX_LEN = 128
TRAIN_BATCH_SIZE = 4  # Reduced to help with memory
VALID_BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Categories and subcategories
category_list = [
    'Online and Social Media Related Crime', 'Online Financial Fraud', 
    'Online Gambling Betting', 'RapeGang Rape RGRSexually Abusive Content', 
    'Any Other Cyber Crime', 'Cyber Attack/Dependent Crimes', 
    'Cryptocurrency Crime', 'Sexually Explicit Act', 'Sexually Obscene material',
    'Hacking Damage to computercomputer system etc', 'Cyber Terrorism',
    'Child Pornography CPChild Sexual Abuse Material CSAM',
    'Online Cyber Trafficking', 'Ransomware', 'Report Unlawful Content', 'Other'
]

subcategory_list = [
    'Cyber Bullying Stalking Sexting', 'Fraud CallVishing', 
    'Online Gambling Betting', 'Online Job Fraud', 'UPI Related Frauds',
    'Internet Banking Related Fraud', 'Profile Hacking Identity Theft',
    'DebitCredit Card FraudSim Swap Fraud', 'EWallet Related Fraud',
    'Data Breach/Theft', 'Cheating by Impersonation',
    'Denial of Service (DoS)/Distributed Denial of Service (DDOS) attacks',
    'FakeImpersonating Profile', 'Cryptocurrency Fraud', 'Malware Attack',
    'Business Email CompromiseEmail Takeover', 'Email Hacking',
    'Hacking/Defacement', 'Unauthorised AccessData Breach', 'SQL Injection',
    'Provocative Speech for unlawful acts', 'Ransomware Attack',
    'Cyber Terrorism', 'Tampering with computer source documents',
    'DematDepository Fraud', 'Online Trafficking', 'Online Matrimonial Fraud',
    'Website DefacementHacking', 'Damage to computer computer systems etc',
    'Impersonating Email', 'EMail Phishing', 'Ransomware',
    'Intimidating Email', 'Against Interest of sovereignty or integrity of India',
    'Other'
]

def normalize_text(text):
    """Normalize text by removing extra spaces and standardizing format"""
    if pd.isna(text):
        return "Other"
    text = str(text).strip()
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    # Remove special characters that might cause matching issues
    text = text.replace('/', ' ')
    text = text.replace('  ', ' ')  # Replace double spaces again after slash removal
    return text

def get_category_index(category, category_mapping):
    """Get index of category with fuzzy matching"""
    normalized_category = normalize_text(category)
    if normalized_category in category_mapping:
        return category_mapping[normalized_category]
    # Return index of 'Other' category as fallback
    return category_mapping.get('Other', len(category_mapping) - 1)

# Create normalized category and subcategory mappings
category_mapping = {normalize_text(cat): idx for idx, cat in enumerate(category_list)}
subcategory_mapping = {normalize_text(subcat): idx for idx, subcat in enumerate(subcategory_list)}


class CyberCrimeDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = df['crimeaditionalinfo'].astype(str)
        
        # Use mapping functions for categories and subcategories
        self.categories = df['category'].apply(lambda x: get_category_index(x, category_mapping))
        self.subcategories = df['sub_category'].apply(lambda x: get_category_index(x, subcategory_mapping))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        category = self.categories.iloc[idx]
        subcategory = self.subcategories.iloc[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'category': torch.tensor(category, dtype=torch.long),
            'subcategory': torch.tensor(subcategory, dtype=torch.long)
        }

class HierarchicalClassifier(nn.Module):
    def __init__(self, num_categories, num_subcategories):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('distilroberta-base')
        self.dropout = nn.Dropout(0.3)
        self.category_classifier = nn.Linear(768, num_categories)
        self.subcategory_classifier = nn.Linear(768, num_subcategories)
        self.bn1 = nn.BatchNorm1d(768)
        self.bn2 = nn.BatchNorm1d(768)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        category_features = self.bn1(pooled_output)
        category_features = self.dropout(category_features)
        category_output = self.category_classifier(category_features)
        
        subcategory_features = self.bn2(pooled_output)
        subcategory_features = self.dropout(subcategory_features)
        subcategory_output = self.subcategory_classifier(subcategory_features)
        
        return category_output, subcategory_output

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        category_targets = batch['category'].to(device)
        subcategory_targets = batch['subcategory'].to(device)
        
        # Regular forward pass without autocast
        category_outputs, subcategory_outputs = model(input_ids, attention_mask)
        category_loss = nn.CrossEntropyLoss()(category_outputs, category_targets)
        subcategory_loss = nn.CrossEntropyLoss()(subcategory_outputs, subcategory_targets)
        loss = category_loss + subcategory_loss
        
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evaluate(model, data_loader, device):
    model.eval()
    category_predictions = []
    subcategory_predictions = []
    category_targets = []
    subcategory_targets = []
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc='Evaluating')
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            category_target = batch['category'].to(device)
            subcategory_target = batch['subcategory'].to(device)
            
            category_output, subcategory_output = model(input_ids, attention_mask)
            
            # Calculate loss
            category_loss = nn.CrossEntropyLoss()(category_output, category_target)
            subcategory_loss = nn.CrossEntropyLoss()(subcategory_output, subcategory_target)
            loss = category_loss + subcategory_loss
            total_loss += loss.item()
            
            # Store predictions and targets
            category_predictions.extend(torch.argmax(category_output, dim=1).cpu().numpy())
            subcategory_predictions.extend(torch.argmax(subcategory_output, dim=1).cpu().numpy())
            category_targets.extend(category_target.cpu().numpy())
            subcategory_targets.extend(subcategory_target.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(data_loader)
    
    return {
        'loss': avg_loss,
        'category_report': classification_report(category_targets, category_predictions, target_names=category_list),
        'subcategory_report': classification_report(subcategory_targets, subcategory_predictions, target_names=subcategory_list)
    }

def save_model(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def main():
    # Load data
    from google.colab import drive
    drive.mount('/content/drive')
    print("Loading data...")
    df = pd.read_csv('/content/drive/My Drive/train.csv')
    # Clean categories and subcategories
    df['category'] = df['category'].apply(normalize_text)
    df['sub_category'] = df['sub_category'].apply(normalize_text)
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)
    
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Initialize tokenizer and create datasets
    print("Initializing tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    
    print("Creating datasets...")
    train_dataset = CyberCrimeDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = CyberCrimeDataset(val_df, tokenizer, MAX_LEN)
    test_dataset = CyberCrimeDataset(test_df, tokenizer, MAX_LEN)
    
    # Create data loaders with adjusted batch sizes
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model and optimizer
    print("Initializing model...")
    model = HierarchicalClassifier(
        num_categories=len(category_list),
        num_subcategories=len(subcategory_list)
    )
    model = model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=EPOCHS)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        print(f'Training Loss: {train_loss:.4f}')
        
        # Validation
        print("\nRunning validation...")
        val_results = evaluate(model, val_loader, DEVICE)
        print(f'Validation Loss: {val_results["loss"]:.4f}')
        
        print(f'\nCategory Classification Report:\n{val_results["category_report"]}')
        print(f'\nSubcategory Classification Report:\n{val_results["subcategory_report"]}')
        
        # Save best model
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            print(f"New best model found! Saving checkpoint...")
            save_model(model, optimizer, epoch, val_results['loss'], 
                     '/content/drive/My Drive/best_model.pt')
    
    # Final evaluation on test set
    print("\nTraining completed. Loading best model for final evaluation...")
    model, optimizer, _, _ = load_model(model, optimizer, 
                                      '/content/drive/My Drive/best_model.pt')
    
    print("\nRunning final evaluation on test set...")
    test_results = evaluate(model, test_loader, DEVICE)
    
    print("\nFinal Test Results:")
    print(f'Test Loss: {test_results["loss"]:.4f}')
    print(f'\nCategory Classification Report:\n{test_results["category_report"]}')
    print(f'\nSubcategory Classification Report:\n{test_results["subcategory_report"]}')

if __name__ == "__main__":
    main()