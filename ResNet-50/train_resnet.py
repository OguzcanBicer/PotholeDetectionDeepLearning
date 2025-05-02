import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import time
import os
import csv

# 📌 Ayarlar
model_name = "resnet50"
epochs = 10
batch_size = 32
image_size = 224
num_classes = 2

# 🔁 Verileri dönüştür
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 📁 Dataset yükle
train_dataset = ImageFolder('../resnet_dataset/train', transform=transform)
val_dataset = ImageFolder('../resnet_dataset/val', transform=transform)
test_dataset = ImageFolder('../resnet_dataset/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 📦 Modeli yükle
model = resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 🎯 Loss ve optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ⏱ Eğitim süresi ölçümü
start_time = time.time()

# 🏋️ Eğitim döngüsü
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

train_time = time.time() - start_time
print(f"\n🕒 Training Time: {train_time:.2f} seconds")

# 💾 Model kaydet ve boyut ölç
torch.save(model.state_dict(), f"{model_name}.pt")
model_size = os.path.getsize(f"{model_name}.pt") / (1024 * 1024)
print(f"💾 Model Size: {model_size:.2f} MB")

# 🧠 Parametre sayısı
total_params = sum(p.numel() for p in model.parameters())
print(f"🧠 Total Parameters: {total_params:,}")

# 📊 Validation Accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
val_accuracy = 100 * correct / total
print(f"\n✅ Validation Accuracy: {val_accuracy:.2f}%")

# 🧪 Test doğruluğu + sınıf bazlı analiz
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
test_accuracy = 100 * correct / total
print(f"\n✅ Test Accuracy: {test_accuracy:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
print("\n📊 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ⚡ Inference süresi (tek görsel)
sample = next(iter(test_loader))[0][0].unsqueeze(0).to(device)
start = time.time()
_ = model(sample)
inference_time = time.time() - start
print(f"⚡ Inference Time (1 image): {inference_time:.4f} seconds")

# 📤 Metrikleri CSV'ye yaz
with open("model_metrics.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        model_name,
        f"{train_time:.2f}",
        f"{model_size:.2f}",
        total_params,
        f"{val_accuracy:.2f}",
        f"{test_accuracy:.2f}",
        f"{inference_time:.4f}"
    ])
