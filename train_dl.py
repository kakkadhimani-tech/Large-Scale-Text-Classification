import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' 
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pyarrow.parquet as pq
from utils import clean_text
from model import CustomTextClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_deep_learning_model():
    print("1. Loading 1 Lakh Rows for Training...")
    parquet_file = pq.ParquetFile('../dataset_10M.parquet')
    df = next(parquet_file.iter_batches(batch_size=100000)).to_pandas() 
    df['CLEAN_DATA'] = df['DATA'].apply(clean_text)

    print("2. Vectorization (Optimizing Memory...)")
    vectorizer = TfidfVectorizer(max_features=20000)
    # Magic Trick: astype('float32') saves 50% RAM!
    X = vectorizer.fit_transform(df['CLEAN_DATA']).astype('float32').toarray()
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['TOPIC'])
    num_classes = len(label_encoder.classes_) 

    print("3. Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Free up memory immediately
    del X 

    # Tensors mein badalna
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Model Initialize
    model = CustomTextClassifier(input_dim=20000, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("\n4. Starting Final Training Loop...")
    epochs = 30
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step() 

        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f} - LR: {scheduler.get_last_lr()[0]}")

    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f"\nFinal Deep Learning Model Accuracy: {accuracy*100:.2f}%")

        # Detailed Report
        print("\nClassification Report:")
        print(classification_report(y_test, predicted.cpu(), target_names=label_encoder.classes_))
        
        print("\nSaving Model Weights...")
        torch.save(model.state_dict(), 'model_weights.pth')
        print(" Model Saved Successfully! Task 100% Complete!")

# Ye hamesha aakhiri line honi chahiye!
if __name__ == "__main__":
    train_deep_learning_model()
