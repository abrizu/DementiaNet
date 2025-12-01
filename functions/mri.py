import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report

# ensure that we run cuda operations on gpu, otherwise training rate will be compressed to cpu only

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
directory = "../Data"
torch.backends.cudnn.benchmark = True 

# -------------------
# hyperparameters
# -------------------

transform_size = 128
batch_size = 64
num_workers = os.cpu_count() or 0
epoch_count = 5
learn_rate = 1e-4
split_rate = 0.8
dropout_rate = 0.4

# -------------------
# data preprocessing
# -------------------

# two different transforms for data augmentation

transform_gen = transforms.Compose([ # generic transform (no augmentation) (not used)
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((transform_size, transform_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_aug = transforms.Compose([ # augmented transform
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((transform_size, transform_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root=directory, transform=transform_aug) 
class_names = dataset.classes

# split dataset into training and testing sets (80% train, 20% test)
def split_dataset(dataset, split_ratio=0.8): # split_ratio = 0.8 for 80% train | the rest test
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_ds, test_ds

def count_images_per_class(dir, class_names):
    counts = {}
    for class_name in class_names:
        class_path = os.path.join(dir, class_name)
        counts[class_name] = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
    return counts

train_ds, test_ds = split_dataset(dataset, split_ratio=split_rate)

# extract labels
targets = [dataset.targets[i] for i in train_ds.indices]

# smaller classes (moderate dementia) get higher weights so the model pays more attention to them
class_sample_counts = np.bincount(targets)
class_weights = 1.0 / torch.sqrt(torch.tensor(class_sample_counts, dtype=torch.float).to(device))
class_weights = class_weights / class_weights.sum()

# compute per-sample weights for the sampler
# weighted random sampler: higher weights get more attention, and are sampled more often
samples_weight = [class_weights[label].item() for label in targets]
sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

# DataLoaders (we add weighted sample so the model can see added weights to minority classes)
train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# -------------------
# CNN Model
# -------------------

class MRI_CNN(nn.Module):
    def __init__(self, num_classes=4): # 4 classes: non-demented, very mild, mild, moderate
        super().__init__()

        # CNN architecture:
        # batch norm -> conv1 -> relu -> maxpool -> conv2 -> relu -> maxpool -> flatten -> fc1 -> relu -> dropout -> fc2
        # 2 conv layers with increasing channels (16, 32) and 2 fully connected layers (128, num_classes)
        # allows for detailed feature extraction while minimizing overfitting via overly complex models 

        # by making a simpler model, reduce training time and improve generalization on unseen data preventing any sort of memorization

        self.norm = nn.BatchNorm2d(1) # batchnorm normalizes input data
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # conv1 = 1 input channel (grayscale), 16 output channels, 3x3 kernel
        self.pool1  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # conv2 = 16 input channels, 32 output channels, 3x3 kernel
        self.pool2 = nn.MaxPool2d(2,2)

        self.relu  = nn.ReLU() # relu 
        self.drop  = nn.Dropout(dropout_rate)
        
        self.fc1   = nn.Linear(32 * 32 * 32, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        # input batch normalization: stabilizes learning process
        # conv layers with relu and maxpool: progressively reduce spatial dimensions while increasing feature depth
        # flatten into single vector
        # dropout and fully connected layers for classification

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
    model.to(device) # move model to gpu if available
    criterion = nn.CrossEntropyLoss(weight=None) # use a cross entropy loss function for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time() # track training time
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(images) # forward pass

            loss = criterion(outputs, labels) # backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    end_time = time.time()
    print(f"\nTraining finished in {(end_time-start_time)/60:.2f} minutes")

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
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))


# -------------------
# visualization
# -------------------

# generated by chatgpt

def plot_sample_predictions(model, data_loader, class_names, num_images=6):
    model.eval()  # set model to evaluation mode

    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, 3, i+1)
        img = images[i].cpu().squeeze(0) * 0.5 + 0.5  # unnormalize
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {class_names[preds[i].item()]}\nTrue: {class_names[labels[i].item()]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# -------------------
# run
# -------------------

def main():

    # print device info, ensure that gpu is primary processor
    print(f"Using device: {device} | "
    + f"Active: {torch.cuda.is_available()} | "
    + f"Version: {torch.__version__} | "
    + f"Count: {torch.cuda.device_count()} | "
    + f"Name: {torch.cuda.get_device_name(0)}"
    )

    image_counts = count_images_per_class(directory, class_names)
    print("Image counts per class:", image_counts)

    print("\nTraining model...")
    model = MRI_CNN(num_classes=len(class_names))
    train_model(model, train_loader, test_loader, num_epochs=epoch_count, lr=learn_rate)

    # visualize some predictions from test set checking overall acc
    plot_sample_predictions(model, test_loader, class_names, num_images=6)

if __name__ == "__main__":
    main()