# RT-DETR POTHOLE DETECTION - TAM EĞİTİM KODU
# Dataset: Mendeley Pothole Dataset

import os
from ultralytics import RTDETR
import torch

# 1- YAML dosyası
yaml_content = """
# Pothole Detection Dataset Config
path: /content/drive/MyDrive/Dataset/Dataset_Mendeley
train: images/train
val: images/val

# Sınıf sayısı
nc: 1

# Sınıf isimleri
names:
  0: pothole
"""

# YAML kaydet
with open('pothole.yaml', 'w') as f:
    f.write(yaml_content)
print("✓ pothole.yaml oluşturuldu")

# 2-Model 
model = RTDETR('rtdetr-x.pt')

# 3- Eğitim Parametreleri
# Toplam Görsel: 3114, Çukur Sayısı: 6887, Ort. Çukur/Görsel: 2.49
# Nesne Boyutu: Çoğunlukla 50-200px, Aspect Ratio: ~1.0-1.5

results = model.train(
    data='pothole.yaml',
    epochs=150,
    imgsz=640,                    
    batch=16,                     
    
    # Optimizer
    optimizer='AdamW',
    lr0=0.00005,                   
    lrf=0.01,                     
    weight_decay=0.0001,
    warmup_epochs=3,
    
    # Augmentation (Küçük nesneler için optimize)
    augment=True,
    mosaic=1.0,                   
    mixup=0.1,                    
    copy_paste=0.1,              
    scale=0.5,                    
    fliplr=0.5,                   
    

    patience=40,                 
    save_period=10,               
    device=0,                     
    workers=4,
    project='pothole_rtdetr',
    name='exp1',
    exist_ok=True,
    
    # Performans
    amp=True,                 
    cache=True,                  
)

print("\n✓ Eğitim tamamlandı!")

# 4- Best model yükleme ve validation.
best_model = RTDETR('/content/runs/detect/pothole_rtdetr/exp1/weights/best.pt')
# Validation metrikleri
metrics = best_model.val(data='pothole.yaml')
print(f"\n--- Validation Sonuçları ---")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

# 5-Test-inference
def predict_image(model, image_path, conf=0.25):
    """Tek bir görsel üzerinde tahmin yap"""
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=0.45,
        save=True,
        project='pothole_rtdetr',
        name='predictions'
    )
    return results

print("\n✓ Modelleme tamam, kod çalışmaya hazır!")