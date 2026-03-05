import os
import torchvision
import cv2
import torch
import argparse
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import functional as F

# 1-Augmentation
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, image, target):
        if np.random.random() < self.prob:
            _, height, width = image.shape
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target

class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.random() < self.prob:
            _, height, width = image.shape
            image = image.flip(-2) 
            
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
        return image, target

class HeavyAugmentation:
    def __init__(self):
        self.color = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05)
        self.autocontrast = T.RandomAutocontrast(p=0.2) 
        
        self.sharp = T.RandomAdjustSharpness(sharpness_factor=2, p=0.4)
        self.blur = T.RandomApply([T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.5))], p=0.2)
        
        self.erase = T.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0)

    def __call__(self, image, target):
        image = self.color(image)
        image = self.autocontrast(image)
        image = self.blur(image)
        image = self.sharp(image)
        image = self.erase(image) 
        return image, target

def get_transform(train):
    transforms = []
    if train:
        transforms.append(HeavyAugmentation())
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))
    return Compose(transforms)



# 2-Dataset ve temizlik
class PotholeDataset(Dataset):
    def __init__(self, root, split="train", size=1280, transforms=None, min_box_size=0, max_samples=None):
        self.root = root
        self.size = size
        self.transforms = transforms
        self.min_box_size = min_box_size
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        
        self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        

        if max_samples is not None and len(self.img_names) > max_samples:
            import random
            random.seed(42)
            random.shuffle(self.img_names)
            self.img_names = self.img_names[:max_samples]
            print(f"[{split.upper()}] LİMİTLENDİ! Rastgele {len(self.img_names)} resim seçildi.")
        else:
            print(f"[{split.upper()}] Toplam Resim (Dolu + Boş): {len(self.img_names)}")

    def __len__(self):
        return len(self.img_names)

    def _yolo_to_pascal(self, yolo_bbox, w, h):
        c, x, y, bw, bh = yolo_bbox
        xmin = (x - bw / 2) * w
        ymin = (y - bh / 2) * h
        xmax = (x + bw / 2) * w
        ymax = (y + bh / 2) * h
        
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lbl_path = os.path.join(self.lbl_dir, os.path.splitext(self.img_names[idx])[0] + ".txt")
        
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_tensor = F.to_tensor(img)

        boxes = []
        labels = []
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.split()))
                    if len(parts) < 5: continue
                    
                    class_id = int(parts[0])
                    if class_id == 0:
                        mapped_label = 1
                    else:
                        continue
                    
                    box = self._yolo_to_pascal(parts, w, h)
                    bw_px = box[2] - box[0]
                    bh_px = box[3] - box[1]
                    
                    if bw_px >= self.min_box_size and bh_px >= self.min_box_size:
                        if box[2] > box[0] and box[3] > box[1]:
                            boxes.append(box)
                            labels.append(mapped_label)
        
        target = {}
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

        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target) 
            
        return img_tensor, target

# 3-Model
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import numpy as np

def advanced_dataset_audit(dataset):
    print("--- Veriseti Analizi Başlıyor (Bu işlem birkaç dakika sürebilir) ---")
    stats = []
    

    for i in range(len(dataset)):
        try:
            img, target = dataset[i]
            # img bir tensor ise: [C, H, W]
            h, w = img.shape[1], img.shape[2]
            boxes = target["boxes"]
            
            if len(boxes) == 0:
                continue
                
            for box in boxes:
                box = box.numpy()
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                stats.append({
                    'width': bw,
                    'height': bh,
                    'area': bw * bh,
                    'aspect_ratio': bw / (bh + 1e-6),
                    'center_x': (box[0] + box[2]) / 2 / w,
                    'center_y': (box[1] + box[3]) / 2 / h
                })
        except Exception as e:
            continue
            
        if i % 500 == 0:
            print(f"{i}/{len(dataset)} resim işlendi...")

    if not stats:
        print("Hata: Hiç kutu (bounding box) bulunamadı! Etiket dosyalarınızı kontrol edin.")
        return

    df = pd.DataFrame(stats)
    
  
    plt.clf() 
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.set_style("whitegrid")

    # 3.1-Aspect Ratio Dağılımı
    sns.histplot(df['aspect_ratio'], bins=50, kde=True, ax=axes[0, 0], color='blue')
    axes[0, 0].set_title("1. Çukur Şekil Analizi (En-Boy Oranı)\n(>1.0 Yatay, <1.0 Dikey)", fontsize=12)
    axes[0, 0].axvline(1.0, color='red', linestyle='--')

    # 3.2-Boyut Dağılımı 
    axes[0, 1].scatter(df['width'], df['height'], alpha=0.2, s=10, color='green')
    axes[0, 1].set_xlabel("Genişlik (px)")
    axes[0, 1].set_ylabel("Yükseklik (px)")
    axes[0, 1].set_title("2. Kutu Boyut Dağılımı\n(Çukurlar kaç piksel?)", fontsize=12)

    # 3.3 Konum Isı Haritası -
    hb = axes[1, 0].hexbin(df['center_x'], df['center_y'], gridsize=25, cmap='YlOrRd', mincnt=1)
    fig.colorbar(hb, ax=axes[1, 0])
    axes[1, 0].set_title("3. Çukurlar Resmin Neresinde?\n(Isı Haritası)", fontsize=12)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].invert_yaxis() 

    # 3.4. Alan (Area) Dağılımı
    sns.histplot(df['area'], bins=50, kde=True, ax=axes[1, 1], color='purple')
    axes[1, 1].set_title("4. Kutuların Alan Dağılımı (Piksel²)\n(Küçük nesne yoğunluğu)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("advanced_analysis.png", dpi=300)
    plt.show()
    print("\n--- Analiz Tamamlandı! 'advanced_analysis.png' dosyası kaydedildi. ---")
    
 
    print("\nKRİTİK İSTATİSTİKLER:")
    print(f"Ortalama En-Boy Oranı: {df['aspect_ratio'].mean():.2f}")
    print(f"Yatay Kutuların Oranı (AR > 1.2): %{len(df[df['aspect_ratio'] > 1.2])/len(df)*100:.1f}")
    print(f"Küçük Nesnelerin Oranı (Alan < 1024 px²): %{len(df[df['area'] < 1024])/len(df)*100:.1f}")
    print("VERİSETİ ANALİZİ BİTTİ")

def get_model_resnet50(num_classes):
    """Veri analizine göre optimize edilmiş model (median: 73x30, mean: 122x79)"""
    
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights, 
        min_size=800,      
        max_size=1333,
        trainable_backbone_layers=3  
    )
    

    """anchor_generator = AnchorGenerator(
        sizes=(
            (16, 32, 40),      # P2
            (48, 64, 80),      # P3
            (112, 128, 160),   # P4
            (256, 320, 384),   # P5
            (512, 768, 1024)   # P6
        ),
        aspect_ratios=((0.5, 1.0, 1.5, 2.0, 3.0),) * 5
    )"""
    anchor_generator = AnchorGenerator(
    sizes=(
        (16, 24, 32),      # Daha küçük çukurlar için (Recall odaklı)
        (64, 78, 96),      
        (128, 160, 192),   
        (256, 320, 384),   
        (512, 604, 700)
    ),
    # K-Means'den gelen [0.93, 1.55, 2.6] değerlerine göre.
    aspect_ratios=((0.9, 1.5, 2.6),) * 5 
    )
    
    model.rpn.anchor_generator = anchor_generator
    
    # RPN Head'i GÜNCELLE (anchor sayısı değişti)
    num_anchors = anchor_generator.num_anchors_per_location()[0]  
    model.rpn.head = torchvision.models.detection.rpn.RPNHead(
        in_channels=256,  
        num_anchors=num_anchors,
        conv_depth=2
    )
    
    # Box Predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Küçük nesneler için ek optimizasyon
    model.roi_heads.box_detections_per_img = 500 
    model.roi_heads.box_score_thresh = 0.01     
    model.rpn._post_nms_top_n['testing'] = 2000
    
    return model

    
def save_error_viz(images, outputs, targets, epoch, folder="error_analysis"):
    os.makedirs(folder, exist_ok=True)
    for i, output in enumerate(outputs):
        img = (images[i].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8).copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for box in targets[i]['boxes']:
            b = box.cpu().numpy().astype(int)
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
        for box, score in zip(output['boxes'], output['scores']):
            if score > 0.45: 
                b = box.cpu().numpy().astype(int)
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cv2.putText(img, f"{score:.2f}", (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imwrite(f"{folder}/epoch_{epoch}_sample_{i}.jpg", img)

def evaluate(model, dataloader, device, iou_thresh=0.45): # IoU thresh gevşettim
    model.eval()
    conf_threshs = np.linspace(0.05, 0.95, 10)
    results = []
    with torch.no_grad():
        for conf in conf_threshs:
            tp, fp, fn = 0, 0, 0
            for images, targets in dataloader:
                images_gpu = [img.to(device) for img in images]
                outputs = model(images_gpu)
                for i, output in enumerate(outputs):
                    gt_boxes = targets[i]['boxes'].to(device)
                    pred_boxes = output['boxes'][output['scores'] > conf]
                    if len(gt_boxes) == 0:
                        fp += len(pred_boxes); continue
                    if len(pred_boxes) == 0:
                        fn += len(gt_boxes); continue
                    iou_matrix = torchvision.ops.box_iou(gt_boxes, pred_boxes)
                    matched_gt = set()
                    for p_idx in range(len(pred_boxes)):
                        best_iou, best_gt = iou_matrix[:, p_idx].max(0)
                        if best_iou >= iou_thresh and best_gt.item() not in matched_gt:
                            tp += 1
                            matched_gt.add(best_gt.item())
                        else:
                            fp += 1
                    fn += (len(gt_boxes) - len(matched_gt))
            p = tp / (tp + fp + 1e-6)
            r = tp / (tp + fn + 1e-6)
            f1 = 2 * p * r / (p + r + 1e-6)
            results.append((conf, p, r, f1))
    print("\nConf | Prec | Rec | F1")
    for res in results:
        print(f"{res[0]:.2f} | {res[1]:.2f} | {res[2]:.2f} | {res[3]:.2f}")
    print("Epoch tahminleri yapıldı")
    return max([res[3] for res in results]) if results else 0.0

# 4-Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001) 
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iou-thresh", type=float, default=0.45) 
    args = parser.parse_args()

    train_ds = PotholeDataset(args.data_root, "train", size = 1280, transforms=get_transform(train=True), min_box_size=0, max_samples=None)
    val_ds = PotholeDataset(args.data_root, "val", size = 1280, transforms=get_transform(train=False), min_box_size=0, max_samples=400)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    #advanced_dataset_audit(train_ds)
    model = get_model_resnet50(num_classes=2).to(args.device)
    swa_model = AveragedModel(model).to(args.device)



    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)    
    
    swa_start = 75 
    swa_scheduler = SWALR(optimizer, swa_lr=0.0001)

    scaler = torch.amp.GradScaler('cuda')
    
    best_f1 = 0.0

    print(f"--- FİNAL EĞİTİM BAŞLIYOR (SWA Başlangıcı: {swa_start}. Epoch) ---")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for i , (images, targets) in enumerate(train_loader):
                  
            images = [img.to(args.device) for img in images]
            targets = [{k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                losses = (
                          loss_dict['loss_classifier'] * 1.0 +
                          loss_dict['loss_box_reg'] * 1.5 +
                          loss_dict['loss_objectness'] * 3.0 + # Nesneyi fark etme cezası artırıldı
                          loss_dict['loss_rpn_box_reg'] * 1.0
                      )

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += losses.item()
        
        if epoch >= swa_start:
            swa_model.update_parameters(model) 
            swa_scheduler.step()
        else:
            lr_scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(train_loader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{epoch+1} Epoch tahminleri yapılıyor...")

        eval_model = swa_model if epoch >= swa_start else model
        current_f1 = evaluate(eval_model, val_loader, args.device, iou_thresh=args.iou_thresh)
        
        SAVE_PATH = "/content/drive/MyDrive/Dataset/models" 
        os.makedirs(SAVE_PATH, exist_ok=True)
        if current_f1 > best_f1:
            best_f1 = current_f1
            save_name = "final_model_swa.pth" if epoch >= swa_start else "final_model.pth"
            torch.save(eval_model.state_dict(), save_name)
            print(f"--- New Record Model! F1: {current_f1:.4f} ---")
            
    print("--- SWA Final İşlemleri: Batch Norm Güncelleniyor ---")
    torch.save(swa_model.state_dict(), "_final_champion.pth")

    print(f"Maraton Bitti. En iyi F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()