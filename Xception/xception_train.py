import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import time
import os
import timm
import csv

# 📌 Ayarlar
model_name = "xception"
num_classes = 2
epochs = 10
batch_size = 32
image_size = 299  # Xception için ideal boyut
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔁 Dönüştürme işlemleri
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 📁 Datasetleri yükle
train_dataset = datasets.ImageFolder("../resnet_dataset/train", transform=transform)
val_dataset = datasets.ImageFolder("../resnet_dataset/val", transform=transform)
test_dataset = datasets.ImageFolder("../resnet_dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 📦 Modeli oluştur
model = timm.create_model(model_name, pretrained=True)

# 🔧 Sınıf sayısına göre son katmanı değiştir
if hasattr(model, 'fc'):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
elif hasattr(model, 'classifier'):
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
elif hasattr(model, 'head'):
    model.head = nn.Linear(model.head.in_features, num_classes)
else:
    raise ValueError("Son katman bulunamadı, manuel ayarlama gerekebilir.")

model = model.to(device)

# 🎯 Loss ve optimizer
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
print(f"\n🕒 Total Training Time: {train_time:.2f} seconds")

# 💾 Modeli kaydet
model_path = f"{model_name}.pt"
torch.save(model.state_dict(), model_path)
model_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"💾 Model Size: {model_size:.2f} MB")

# 🧠 Parametre sayıları
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"🧠 Total Parameters: {total_params:,}")
print(f"🔧 Trainable Parameters: {trainable_params:,}")

# 📊 Validation Accuracy
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
print(f"\n✅ Validation Accuracy: {val_acc:.2f}%")

# 🧪 Test Accuracy ve analiz
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
print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
print("\n📊 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ⚡ Inference süresi (tek görsel)
sample_image = next(iter(test_loader))[0][0].unsqueeze(0).to(device)
start = time.time()
_ = model(sample_image)
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
        trainable_params,
        f"{val_acc:.2f}",
        f"{test_acc:.2f}",
        f"{inference_time:.4f}"
    ])
