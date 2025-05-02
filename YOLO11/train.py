from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")  # Veya yolov8n.pt
    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=300,
        batch=16
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Opsiyonel ama güvenli
    main()