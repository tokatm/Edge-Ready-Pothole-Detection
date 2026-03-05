import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np

class PotholeDataset(Dataset):
    def __init__(self, root, split="train", size=None, transforms=None, min_box_size=10):
        # Not: size=None yaptık, çünkü görselleştirmede resize yapmasak daha net görürüz.
        self.root = root
        self.transforms = transforms
        self.min_box_size = min_box_size
        
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        
        self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG'))])
        
        print(f"[{split.upper()}] Yüklenen Resim Sayısı: {len(self.img_names)}")

    def __len__(self):
        return len(self.img_names)

    def _yolo_to_pascal(self, yolo_bbox, w, h):
        c, x, y, bw, bh = yolo_bbox
        
        xmin = (x - bw / 2) * w
        ymin = (y - bh / 2) * h
        xmax = (x + bw / 2) * w
        ymax = (y + bh / 2) * h
        
        # Clamp: Resim dışına taşmayı engelle
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(self.lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_tensor = F.to_tensor(img) #

        boxes = []
        labels = []
        
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.split()))
                    if len(parts) < 5: continue 
                    
                    # Sınıf ID Kontrolü (0 -> 1 Mapping)
                    class_id = int(parts[0])
                    if class_id == 0:
                        mapped_label = 1 
                    else:
                        continue 
                    
                    box = self._yolo_to_pascal(parts, w, h)
                    
                    # Küçük Kutu Filtresi(Piksel)
                    bw_px = box[2] - box[0]
                    bh_px = box[3] - box[1]
                    
                    if bw_px >= self.min_box_size and bh_px >= self.min_box_size:
                        if box[2] > box[0] and box[3] > box[1]:
                            boxes.append(box)
                            labels.append(mapped_label)
        
        target = {}
        target["image_id"] = torch.tensor([idx])
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, target, img_name

def visualize_local(dataset, count=3):
    print(f"\n>>> Toplam {count} adet Rastgele Örnek Gösteriliyor...")
    
    indices = [random.randint(0, len(dataset) - 1) for _ in range(count)]
    
    for i, idx in enumerate(indices):
        img_tensor, target, filename = dataset[idx]
        
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.imshow(img_np)
        
        box_count = len(target['boxes'])
        status = "POZİTİF (Çukur Var)" if box_count > 0 else "NEGATİF (Temiz Yol)"
        color_status = "green" if box_count > 0 else "blue"
        
        print(f"[{i+1}/{count}] Dosya: {filename} | Durum: {status} | Kutu Sayısı: {box_count}")
        
        if box_count > 0:
            for j, box in enumerate(target['boxes']):
                xmin, ymin, xmax, ymax = box.numpy()
                width = xmax - xmin
                height = ymax - ymin
                
                # Kutuyu Çiz (Kırmızı)
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin - 5, "Pothole", color='yellow', fontsize=10, weight='bold', backgroundcolor='red')

        plt.title(f"{filename} - {status}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    local_root_path = "/Users/mustafa/Library/Mobile Documents/com~apple~CloudDocs/Computer_Science/09-Makalelerim/6-ADAS/dataset-2"
    
    try:
        dataset = PotholeDataset(root=local_root_path, split="train")
        
        # kontrol
        visualize_local(dataset, count=5)
        
    except FileNotFoundError:
        print("HATA: Klasör yolu bulunamadı!")
        print(f"Aranan yol: {local_root_path}")
        print("Lütfen yolun doğru olduğundan ve klasör izinlerinden emin ol.")