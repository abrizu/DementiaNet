import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report
import torch.optim as optim
import time
import os
# import kaggle

# ensure that we run cuda operations on gpu, otherwise training rate is very slow
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | "
    + f"Active: {torch.cuda.is_available()} | "
    + f"Version: {torch.version.cuda} | "
    + f"Count: {torch.cuda.device_count()} | "
    + f"Name: {torch.cuda.get_device_name(0)}"
    )

directory = "./Data"

# -------------------
# Data Preprocessing
# -------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root=directory, transform=transform) 
class_names = dataset.classes

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)  # set to 0 for now
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

# Function to count images per class
def count_images_per_class(dir, class_names):
    counts = {}
    for class_name in class_names:
        class_path = os.path.join(dir, class_name)
        counts[class_name] = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
    return counts

image_counts = count_images_per_class(directory, class_names)
print("Image counts per class:", image_counts)

# -------------------
# CNN Model
# -------------------

class MRI_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.2)
        
        self.fc1   = nn.Linear(64 * 16 * 16, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -------------------
# Training Function
# -------------------
def train_model(model, train_loader, test_loader, num_epochs=10, lr=1e-4):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    print(f"\nTraining finished in {(time.time()-start_time)/60:.2f} minutes")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))

# -------------------
# Run
# -------------------

def main():
    print("\nTraining model...")
    model = MRI_CNN(num_classes=len(class_names))
    train_model(model, train_loader, test_loader, num_epochs=3, lr=1e-4)

if __name__ == "__main__":
    main()