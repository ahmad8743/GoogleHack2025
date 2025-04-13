import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from enums import NUM_CLASSES

def main():
    data = pd.read_pickle('hand_landmarks.pkl.gz')
    X = np.stack(data['landmarks'].apply(flatten_landmarks))
    y = data['class'].astype('category').cat.codes.values
    labels = data['class'].astype('category').cat.categories.to_list()
    with open('labels.json', 'w') as f:
        json.dump(labels, f, indent=4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = HandLandmarkDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = HandModelNN(input_dim=63, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, criterion, optimizer, device, epochs=10000)

    test_dataset = HandLandmarkDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    evaluate_model(model, test_loader, device)

    torch.save(model.state_dict(), 'model.pth')

class HandLandmarkDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class HandModelNN(nn.Module):
    def __init__(self, input_dim=63, num_classes=NUM_CLASSES):
        super(HandModelNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.model(x)

def train_model(model, dataloader, criterion, optimizer, device, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

def evaluate_model(model, dataloader, device):
    model.eval()  # Set model to eval mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

def flatten_landmarks(landmarks):
    return np.array(landmarks).flatten()

def flatten_landmarks_2(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

if __name__ == '__main__':
    main()

