from ultralytics import YOLO

# Model
model = YOLO('yolo11m.pt')  # medium (m/l/x seç)

# Eğitim
results = model.train(
    data='/content/pothole.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    lr0=0.0001,
    patience=20,
    device=0
)

# Validation
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")


# yaml
path: /content/drive/MyDrive/Dataset/Dataset_Mendeley
train: images/train
val: images/val

nc: 1
names: ['pothole']