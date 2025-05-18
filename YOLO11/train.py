from ultralytics import YOLO

def main():
    model = YOLO("yolo11m.pt")  
    results = model.train(
        data="dataset/data.yaml",
        epochs=25,
        imgsz=300,
        batch=16,
        pretrained=False
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Opsiyonel ama güvenli
    main()