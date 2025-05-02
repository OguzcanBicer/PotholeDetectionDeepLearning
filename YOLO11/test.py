from ultralytics import YOLO
import os

model = YOLO("runs/detect/train2/weights/best.pt")
model.predict(source="dataset/images/test", save=True)

# PyTorch model nesnesine eriş
torch_model = model.model

# Toplam parametre ve eğitilebilir parametre sayısı
total_params = sum(p.numel() for p in torch_model.parameters())
trainable_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)

print(f"🧠 Total Parameters: {total_params:,}")
print(f"🔧 Trainable Parameters: {trainable_params:,}")

model_path = "runs/detect/train2/weights/best.pt"
model_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"💾 Model Size: {model_size:.2f} MB")