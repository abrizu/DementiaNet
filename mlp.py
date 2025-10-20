import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# === Load combined embeddings ===
combined_dict = torch.load("C:/Users/ahmed/Downloads/combined_embeddings.pt")

# Convert dict values to tensors for dataset
features = torch.stack([v[0] for v in combined_dict.values()])  # [num_samples, 1536]
labels   = torch.stack([v[1] for v in combined_dict.values()])  # [num_samples]

# Create TensorDataset and DataLoader
dataset = TensorDataset(features, labels)

# Shuffle and split into train/test
num_samples = len(dataset)
train_size = int(0.8 * num_samples)
test_size = num_samples - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === Define MLP ===
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_dim = features.shape[1]  # 768 + 768
num_classes = 2
hidden_dim = 128
model = SimpleMLP(input_dim, hidden_dim, num_classes)

# === Loss and optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Training loop ===
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# === Evaluation ===
x_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

model.eval()
with torch.no_grad():
    outputs = model(x_test)
    _, predicted = torch.max(outputs, dim=1)

accuracy = accuracy_score(y_test, predicted)
print(f'Accuracy: {accuracy*100:.2f}%')


