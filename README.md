# AIAA-hw2

## model.py
```python
# coding=utf-8
"""Models."""

import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,dropout_prob=0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x))))
        x = self.bn2(self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, 256 * 56 * 56)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_prob)
        x = self.fc2(x)
        return x
```

## CNN_Classifier_train.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import time
from models import Net
from torchvision.models import resnet50, ResNet50_Weights


class MyDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.transforms = transform
        self.df = pd.read_csv(csv_file, header=None, skiprows=1)
        self.classes = sorted(self.df[1].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid, label = self.df.iloc[index, :]
        img_list = os.listdir(os.path.join(self.root, f"{vid}"))
        img_list = sorted(img_list)
        img_path = os.path.join(self.root, f"{vid}", img_list[int(len(img_list)/2)])
 
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.classes.index(label)
        return img, label

# You can add data augmentation here
transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally             
            transforms.RandomRotation(10),  # Randomly rotate in [-10°, 10°]
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),  # Transform to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
        ])

device = torch.device('cpu')  # ('cuda' if torch.cuda.is_available() else 'cpu')

resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet = resnet.to(device)

trainval_dataset = MyDataset("video_frames_30fpv_320p", "trainval.csv", transform)
train_data, val_data = train_test_split(trainval_dataset, test_size=0.2, random_state=0)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

unique_categories = trainval_dataset.df[1].unique()
num_class = len(unique_categories)
resnet.fc = nn.Linear(resnet.fc.in_features, num_class)

for epoch in range(5):
    start_time = time.time()
    # TODO: Metrics variables ...
    running_loss_train = 0.0
    running_loss_val = 0.0
    correct_train = 0
    total_train = 0
    correct_val = 0
    total_val = 0
    best_val_accuracy = 0.0
    
    resnet.train()
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # TODO: Training code ...
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss_train += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    resnet.eval()
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            # TODO: Validation code ...
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            outputs = resnet(val_inputs)
            loss = criterion(outputs, val_labels)
            running_loss_val += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += val_labels.size(0)
            correct_val += (predicted == val_labels).sum().item()

    # TODO: save best model
    current_val_accuracy = correct_val / total_val
    if current_val_accuracy > best_val_accuracy:
        best_val_accuracy = current_val_accuracy
        torch.save(resnet.state_dict(), 'model_best.pth')

    # save last model
    torch.save(resnet.state_dict(), 'model_last.pth')

    end_time = time.time()

    # print metrics log
    print('[Epoch %d] Loss (train/val): %.3f/%.3f' % (epoch + 1, running_loss_train/len(train_loader), running_loss_val/len(val_loader)),
        ' Acc (train/val): %.2f%%/%.2f%%' % (100 * correct_train/total_train, 100 * correct_val/total_val),
        ' Epoch Time: %.2f' % (end_time - start_time))
```

## CNN_Classifier_test2csv.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from models import Net
from torchvision.models import resnet50, ResNet50_Weights


device = torch.device('cpu')

class TestDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.transforms = transform
        self.df = pd.read_csv(csv_file, header=None, skiprows=1)
        self.classes = sorted(self.df[1].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid, label = self.df.iloc[index, :]
        img_list = os.listdir(os.path.join(self.root, f"{vid}"))
        img_list = sorted(img_list)
        img_path = os.path.join(self.root, f"{vid}", img_list[int(len(img_list)/2)])
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.classes.index(label)
        return img, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test data
test_dataset = TestDataset("video_frames_30fpv_320p", "test_for_student.csv", transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = nn.Linear(resnet.fc.in_features, 10) # Ensure this matches the number of classes in the model
resnet.to(device)
resnet.load_state_dict(torch.load('model_last.pth'))
resnet.eval()


results = []
with torch.no_grad():
    # TODO: Evaluation result here ...
    for images, video_ids in test_loader:
        images = images.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
# Collect predictions for each video ID
        for vid, pred_class in zip(video_ids, predicted.cpu().numpy()):
            results.append((vid, pred_class))

with open('result.csv', 'w') as f:
    f.write("Id,Category\n")
    for vid, category in results:
        f.write(f"{vid},{category}\n")
```
