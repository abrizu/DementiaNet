import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report
import torch.optim as optim
import time
import os
# import kaggle
import numpy as np
from torch.utils.data import WeightedRandomSampler



# ensure that we run cuda operations on gpu, otherwise training rate will be compressed to cpu only

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
directory = "./Data"
torch.backends.cudnn.benchmark = True 

# -------------------
# hyperparameters
# -------------------

transform_size = 128
num_classes = 4
batch_size = 64
num_workers = os.cpu_count() 
epoch_count = 3
learn_rate = 1e-4


# -------------------
# data preprocessing
# -------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((transform_size, transform_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root=directory, transform=transform) 
class_names = dataset.classes

# split dataset into training and testing sets (80% train, 20% test)
def split_dataset(dataset, split_ratio=0.8): # split_ratio = 0.8 for 80% train | the rest test
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_ds, test_ds

# Function to count images per class
def count_images_per_class(dir, class_names):
    counts = {}
    for class_name in class_names:
        class_path = os.path.join(dir, class_name)
        counts[class_name] = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
    return counts

train_ds, test_ds = split_dataset(dataset, split_ratio=0.8)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# -------------------
# CNN Model
# -------------------

class MRI_CNN(nn.Module):
    def __init__(self, num_classes=4): # 4 classes: non-demented, very mild, mild, moderate
        super().__init__()
        self.norm = nn.BatchNorm2d(1)

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1  = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)    
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.3)
        
        self.fc1   = nn.Linear(32 * 32 * 32, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.pool1(self.relu(self.conv1(x))) # 128x128 -> 64x64 
        x = self.pool2(self.relu(self.conv2(x))) # 64x64 -> 32x32
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

    print(f"Using device: {device} | "
    + f"Active: {torch.cuda.is_available()} | "
    + f"Version: {torch.version.cuda} | "
    + f"Count: {torch.cuda.device_count()} | "
    + f"Name: {torch.cuda.get_device_name(0)}"
    )

    image_counts = count_images_per_class(directory, class_names)
    print("Image counts per class:", image_counts)

    print("\nTraining model...")
    model = MRI_CNN(num_classes=len(class_names))
    train_model(model, train_loader, test_loader, num_epochs=epoch_count, lr=learn_rate)

if __name__ == "__main__":
    main()