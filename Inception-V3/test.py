import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ✅ Cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔁 Veri dönüşümleri
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 için
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 📁 Dataset yükle
train_dataset = datasets.ImageFolder("resnet_dataset/train", transform=transform)
val_dataset = datasets.ImageFolder("resnet_dataset/val", transform=transform)
test_dataset = datasets.ImageFolder("resnet_dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 📦 Model: InceptionV3
model = models.inception_v3(pretrained=True, aux_logits=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 sınıf: pothole / no_pothole
model = model.to(device)

# 🎯 Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 🏋️ Eğitim döngüsü
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, aux_outputs = model(images)
        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux_outputs, labels)
        loss = loss1 + 0.4 * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} - Training Loss: {running_loss:.4f}")

# 📊 Val doğruluğu
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_acc = 100 * correct / total
print(f"\nValidation Accuracy: {val_acc:.2f}%")

# 🧪 Test doğruluğu
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * correct / total
print(f"\nTest Accuracy: {test_acc:.2f}%")

# 📈 Sınıf bazlı analiz
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
