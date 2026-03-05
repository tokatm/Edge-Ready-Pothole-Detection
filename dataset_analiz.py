import os
from PIL import Image
import numpy as np

def analyze_resolution_impact(img_dir, target_min_size=800):
    widths = []
    heights = []
    img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')][:500]
    
    for name in img_names:
        with Image.open(os.path.join(img_dir, name)) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    
    avg_w = np.mean(widths)
    avg_h = np.mean(heights)
    min_side = min(avg_w, avg_h)
    
    scale_factor = target_min_size / min_side
    
    print(f"--- Çözünürlük Analizi ---")
    print(f"Ortalama Çözünürlük: {avg_w:.0f}x{avg_h:.0f}")
    print(f"Ölçekleme Faktörü (Scale): {scale_factor:.2f}")
    
    # Analizdeki median değerleri buraya yazalım (128x102)
    median_w, median_h = 128, 102
    new_w = median_w * scale_factor
    new_h = median_h * scale_factor
    
    print(f"Model işlerken Median Çukur Boyutu: {new_w:.1f}x{new_h:.1f} piksel olacak.")
    
    if new_w < 30 or new_h < 30:
        print("KRİTİK UYARI: Çukurlar çok küçülüyor! min_size artırılmalı.")
    else:
        print("SONUÇ: 800px bu veri seti için güvenli görünüyor.")

# Kullanım:
analyze_resolution_impact("Dataset_Mendeley/images/train")