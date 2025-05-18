import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time
import os
import timm
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# 📌 Ayarlar
model_name = "xception"
num_classes = 2
epochs = 25
batch_size = 32
image_size = 299
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔁 Dönüştürme
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 📁 Dataset
train_dataset = datasets.ImageFolder("../resnet_dataset/train", transform=transform)
val_dataset = datasets.ImageFolder("../resnet_dataset/val", transform=transform)
test_dataset = datasets.ImageFolder("../resnet_dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 📦 Modeli oluştur (sıfırdan)
model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model = model.to(device)

# 🎯 Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ⏱ Eğitim süresi başlat
start_time = time.time()
train_losses = []
val_accuracies = []

# 🏋️ Eğitim döngüsü
for epoch in range(1, epochs + 1):
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
    train_losses.append(total_loss)

    # 🔍 Doğrulama
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch}/{epochs} - Loss: {total_loss:.4f} - Validation Accuracy: {val_acc:.2f}%")

train_time = time.time() - start_time

# 💾 Model kaydet
torch.save(model.state_dict(), f"{model_name}.pt")
model_size = os.path.getsize(f"{model_name}.pt") / (1024 * 1024)
total_params = sum(p.numel() for p in model.parameters())

# 🧪 Test metrikleri
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

# ⚡ Inference süresi
sample_image = next(iter(test_loader))[0][0].unsqueeze(0).to(device)
start = time.time()
_ = model(sample_image)
inference_time = time.time() - start

# 📊 Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {model_name}")
plt.savefig(f"{model_name}_confusion_matrix.png")
plt.close()

# 📈 Eğitim Eğrileri
plt.figure()
plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'{model_name} - Training Loss over Epochs')
plt.grid()
plt.legend()
plt.savefig(f"{model_name}_loss_curve.png")
plt.close()

plt.figure()
plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title(f'{model_name} - Validation Accuracy over Epochs')
plt.grid()
plt.legend()
plt.savefig(f"{model_name}_val_accuracy.png")
plt.close()

# 📊 Sonuçlar
print(f"\n📊 Model: {model_name}")
print(f"🧠 Parametre Sayısı: {total_params:,}")
print(f"💾 Model Boyutu: {model_size:.2f} MB")
print(f"🕒 Eğitim Süresi: {train_time:.2f} saniye")
print(f"⚡ Inference Süresi (1 görüntü): {inference_time:.4f} saniye")
print(f"🎯 Precision: {precision:.4f}")
print(f"🎯 Recall: {recall:.4f}")
print(f"🎯 F1-Score: {f1:.4f}")
print("\n📊 Confusion Matrix:")
print(cm)